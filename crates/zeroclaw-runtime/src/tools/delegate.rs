use crate::agent::loop_::{TOOL_LOOP_SESSION_KEY, run_tool_call_loop};
use crate::agent::prompt::{PromptContext, SystemPromptBuilder};
use crate::observability::traits::{Observer, ObserverEvent, ObserverMetric};
use crate::security::SecurityPolicy;
use crate::security::policy::ToolOperation;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatRequest;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::schema::{
    AliasedAgentConfig, Config, DelegateToolConfig, ModelProviderConfig, RiskProfileConfig,
    RuntimeProfileConfig, SkillBundleConfig,
};
use zeroclaw_log::Instrument as _;
use zeroclaw_memory::Memory;
use zeroclaw_providers::{self, ChatMessage, ModelProvider};

/// Shared neutral base workspace used by [`DelegateTool::profile_grants`] as
/// the `workspace_dir` for BOTH agents' capability-grants policies. Resolving
/// every agent's risk-profile roots against the SAME base means a per-agent
/// workspace jail (each agent's distinct real `workspace_dir`) never reads as
/// a false escalation; the cross-agent FS tiers (`workspace.access`,
/// `unrestricted_filesystem`) are re-applied on top so genuine broadening is
/// still compared.
const GRANT_CMP_WORKSPACE: &str = "/__zc_delegate_grant_cmp__";

fn current_tool_loop_session_key() -> Option<String> {
    TOOL_LOOP_SESSION_KEY.try_with(Clone::clone).ok().flatten()
}

async fn scope_delegate_session_key<F>(session_key: Option<String>, future: F) -> F::Output
where
    F: std::future::Future,
{
    TOOL_LOOP_SESSION_KEY.scope(session_key, future).await
}

/// Serializable result of a background delegate task.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackgroundDelegateResult {
    pub task_id: String,
    pub agent: String,
    pub status: BackgroundTaskStatus,
    pub output: Option<String>,
    pub error: Option<String>,
    pub started_at: String,
    pub finished_at: Option<String>,
}

/// Status of a background delegate task.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundTaskStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Tool that delegates a subtask to a named agent with a different
/// model_provider/model configuration. Enables multi-agent workflows where
/// a primary agent can hand off specialized work (research, coding,
/// summarization) to purpose-built sub-agents.
///
/// Supports three execution modes:
/// - **Synchronous** (default): blocks until the sub-agent completes.
/// - **Background** (`background: true`): spawns the sub-agent in a tokio
///   task and returns a `task_id` immediately.
/// - **Parallel** (`parallel: [...]`): runs multiple agents concurrently
///   and returns all results.
///
/// Background results are persisted to `workspace/delegate_results/{task_id}.json`
/// and can be retrieved via `action: "check_result"`.
pub struct DelegateTool {
    agents: Arc<HashMap<String, AliasedAgentConfig>>,
    security: Arc<SecurityPolicy>,
    /// Global credential (from config.api_key) used when an agent has none set.
    global_credential: Option<String>,
    /// ModelProvider runtime options inherited from root config.
    provider_runtime_options: zeroclaw_providers::ModelProviderRuntimeOptions,
    /// Depth at which this tool instance lives in the delegation chain.
    depth: u32,
    /// Parent tool registry for agentic sub-agents.
    parent_tools: Arc<RwLock<Vec<Arc<dyn Tool>>>>,
    /// Inherited multimodal handling config for sub-agent loops.
    multimodal_config: zeroclaw_config::schema::MultimodalConfig,
    /// Global delegate tool config providing default timeout values.
    delegate_config: DelegateToolConfig,
    /// Workspace directory inherited from the root agent context.
    workspace_dir: PathBuf,
    /// Cancellation token for cascade control of background tasks.
    cancellation_token: CancellationToken,
    /// Optional memory instance for namespace isolation on delegate agents.
    memory: Option<Arc<dyn Memory>>,
    /// nested model provider map for brain resolution.
    providers_models: Arc<HashMap<String, HashMap<String, ModelProviderConfig>>>,
    /// named risk profiles for delegation depth and timeout resolution.
    risk_profiles: Arc<HashMap<String, RiskProfileConfig>>,
    /// named runtime profiles for agentic/tools/iteration resolution.
    runtime_profiles: Arc<HashMap<String, RuntimeProfileConfig>>,
    /// named skill bundles for skills-directory resolution.
    skill_bundles: Arc<HashMap<String, SkillBundleConfig>>,
    /// Optional handle to the loaded root config used to resolve a
    /// per-target `SecurityPolicy` at delegate time. When set, every
    /// delegation validates the target agent's policy as a subset of
    /// the calling agent's via `ensure_no_escalation_beyond` and
    /// inherits the caller's `PerSenderTracker` so action / cost
    /// budgets are shared between caller and delegated runs. When
    /// unset (legacy unit-test constructors), DelegateTool falls back
    /// to using `self.security` for the spawned inner DelegateTool.
    root_config: Option<Arc<Config>>,
    /// Optional observer for emitting per-call `LlmResponse` telemetry from
    /// non-agentic delegate runs (which bypass the agent loop's own emission).
    /// `None` for legacy unit-test constructors — telemetry is simply skipped.
    observer: Option<Arc<dyn Observer>>,
    /// Alias of the agent that owns this DelegateTool. Excluded from the
    /// advertised roster so an agent is never offered itself as a
    /// delegation target. Empty when unset (legacy unit-test constructors).
    caller_alias: String,
}

impl DelegateTool {
    pub fn new(
        agents: HashMap<String, AliasedAgentConfig>,
        global_credential: Option<String>,
        security: Arc<SecurityPolicy>,
    ) -> Self {
        Self::new_with_options(
            agents,
            global_credential,
            security,
            zeroclaw_providers::ModelProviderRuntimeOptions::default(),
        )
    }

    pub fn new_with_options(
        agents: HashMap<String, AliasedAgentConfig>,
        global_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        provider_runtime_options: zeroclaw_providers::ModelProviderRuntimeOptions,
    ) -> Self {
        Self {
            agents: Arc::new(agents),
            security,
            global_credential,
            provider_runtime_options,
            depth: 0,
            parent_tools: Arc::new(RwLock::new(Vec::new())),
            multimodal_config: zeroclaw_config::schema::MultimodalConfig::default(),
            delegate_config: DelegateToolConfig::default(),
            workspace_dir: PathBuf::new(),
            cancellation_token: CancellationToken::new(),
            memory: None,
            providers_models: Arc::new(HashMap::new()),
            risk_profiles: Arc::new(HashMap::new()),
            runtime_profiles: Arc::new(HashMap::new()),
            skill_bundles: Arc::new(HashMap::new()),
            root_config: None,
            observer: None,
            caller_alias: String::new(),
        }
    }

    /// Create a DelegateTool for a sub-agent (with incremented depth).
    /// When sub-agents eventually get their own tool registry, construct
    /// their DelegateTool via this method with `depth: parent.depth + 1`.
    pub fn with_depth(
        agents: HashMap<String, AliasedAgentConfig>,
        global_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        depth: u32,
    ) -> Self {
        Self::with_depth_and_options(
            agents,
            global_credential,
            security,
            depth,
            zeroclaw_providers::ModelProviderRuntimeOptions::default(),
        )
    }

    pub fn with_depth_and_options(
        agents: HashMap<String, AliasedAgentConfig>,
        global_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        depth: u32,
        provider_runtime_options: zeroclaw_providers::ModelProviderRuntimeOptions,
    ) -> Self {
        Self {
            agents: Arc::new(agents),
            security,
            global_credential,
            provider_runtime_options,
            depth,
            parent_tools: Arc::new(RwLock::new(Vec::new())),
            multimodal_config: zeroclaw_config::schema::MultimodalConfig::default(),
            delegate_config: DelegateToolConfig::default(),
            workspace_dir: PathBuf::new(),
            cancellation_token: CancellationToken::new(),
            memory: None,
            providers_models: Arc::new(HashMap::new()),
            risk_profiles: Arc::new(HashMap::new()),
            runtime_profiles: Arc::new(HashMap::new()),
            skill_bundles: Arc::new(HashMap::new()),
            root_config: None,
            observer: None,
            caller_alias: String::new(),
        }
    }

    /// Attach parent tools used to build sub-agent allowlist registries.
    pub fn with_parent_tools(mut self, parent_tools: Arc<RwLock<Vec<Arc<dyn Tool>>>>) -> Self {
        self.parent_tools = parent_tools;
        self
    }

    /// Attach multimodal configuration for sub-agent tool loops.
    pub fn with_multimodal_config(
        mut self,
        config: zeroclaw_config::schema::MultimodalConfig,
    ) -> Self {
        self.multimodal_config = config;
        self
    }

    /// Attach global delegate tool configuration for default timeout values.
    pub fn with_delegate_config(mut self, config: DelegateToolConfig) -> Self {
        self.delegate_config = config;
        self
    }

    /// Return a shared handle to the parent tools list.
    /// Callers can push additional tools (e.g. MCP wrappers) after construction.
    pub fn parent_tools_handle(&self) -> Arc<RwLock<Vec<Arc<dyn Tool>>>> {
        Arc::clone(&self.parent_tools)
    }

    /// Attach the workspace directory for system prompt enrichment.
    pub fn with_workspace_dir(mut self, workspace_dir: PathBuf) -> Self {
        self.workspace_dir = workspace_dir;
        self
    }

    /// Resolve a target sub-agent's workspace dir for identity-file
    /// loading. Delegates to `Config::agent_workspace_dir` so the
    /// per-agent path lives in one place; returns `None` when no
    /// `root_config` is attached (legacy unit-test constructors), which
    /// callers treat as "no identity files to load".
    fn agent_workspace(&self, agent_alias: &str) -> Option<PathBuf> {
        self.root_config
            .as_ref()
            .map(|cfg| cfg.agent_workspace_dir(agent_alias))
    }

    /// Attach a cancellation token for cascade control of background tasks.
    /// When the token is cancelled, all background sub-agents are aborted.
    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = token;
        self
    }

    /// Return the cancellation token for external cascade control.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation_token
    }

    /// Attach memory for namespace isolation on delegate agents.
    pub fn with_memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Attach nested model provider map for brain resolution.
    pub fn with_providers_models(
        mut self,
        m: HashMap<String, HashMap<String, ModelProviderConfig>>,
    ) -> Self {
        self.providers_models = Arc::new(m);
        self
    }

    /// Attach risk profiles for depth/timeout resolution.
    pub fn with_risk_profiles(mut self, m: HashMap<String, RiskProfileConfig>) -> Self {
        self.risk_profiles = Arc::new(m);
        self
    }

    /// Attach runtime profiles for agentic/tools/iteration resolution.
    pub fn with_runtime_profiles(mut self, m: HashMap<String, RuntimeProfileConfig>) -> Self {
        self.runtime_profiles = Arc::new(m);
        self
    }

    /// Attach skill bundles for skills-directory resolution.
    pub fn with_skill_bundles(mut self, m: HashMap<String, SkillBundleConfig>) -> Self {
        self.skill_bundles = Arc::new(m);
        self
    }

    /// Attach the loaded root config so DelegateTool can resolve a
    /// per-target `SecurityPolicy` at delegate time, validate it as a
    /// subset of the caller's policy, and share the caller's
    /// `PerSenderTracker` with the delegated run.
    pub fn with_root_config(mut self, config: Arc<Config>) -> Self {
        self.root_config = Some(config);
        self
    }

    /// Attach the runtime observer so non-agentic delegate calls emit a
    /// per-call `LlmResponse` event (provider/model/duration/tokens).
    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Set the owning agent's alias so it can be excluded from the
    /// advertised delegation roster (an agent must never delegate to
    /// itself).
    pub fn with_caller_alias(mut self, alias: impl Into<String>) -> Self {
        self.caller_alias = alias.into();
        self
    }

    /// Build a `SecurityPolicy` for the delegated target agent, enforcing
    /// the **narrowing-only** delegation invariant: a delegate may run under
    /// a *different* risk profile than the caller ONLY when the target's
    /// profile grants are **no broader than** the caller's (no privilege
    /// escalation). Same-profile delegation is unchanged.
    ///
    /// Gates, in order:
    /// 1. `delegation_policy.permits()` — the operator's on/off switch
    ///    (default `forbidden`). Unchanged.
    /// 2. Cross-profile narrowing ([`Self::cross_profile_decision`]):
    ///    - **Same profile** → allowed verbatim (identical grants; the
    ///      agentic loop's reuse of the caller's tool registry is safe).
    ///    - **Different profile, target not broader** (per
    ///      [`SecurityPolicy::ensure_no_escalation_beyond`] over profile
    ///      *grants*, plus an explicit `allowed_tools` subset check) AND
    ///      **non-agentic** → allowed. The target runs under its OWN
    ///      (narrower) policy; non-agentic delegates have no tool registry,
    ///      so there is no escalation surface.
    ///    - **Different profile, target broader** → refused (escalation).
    ///    - **Different profile, agentic** → refused. The in-process agentic
    ///      delegate loop reuses the caller's `parent_tools`, each bound to
    ///      the caller's `Arc<SecurityPolicy>` at construction; running them
    ///      for a different-profile target would enforce the CALLER's policy,
    ///      not the target's — an escalation. Narrowed *agentic* runs go
    ///      through `spawn_subagent` / a shared profile.
    ///
    /// The returned policy's `tracker` is the caller's `Arc`-shared tracker
    /// so delegated actions count against the caller's `max_actions_per_hour`
    /// / `max_cost_per_day_cents`.
    ///
    /// `Ok(self.security)` (the caller's policy) when `root_config` is `None`
    /// — the legacy unit-test constructors that don't plumb root config.
    ///
    /// The narrowing comparison is over profile *grants* built via
    /// [`SecurityPolicy::from_profiles`] on a shared neutral workspace, NOT
    /// the per-agent resolved policies: [`SecurityPolicy::for_agent`] jails
    /// each agent to its own workspace dir, so sibling agents on the SAME
    /// profile have non-overlapping `allowed_roots` that would otherwise read
    /// as a false escalation.
    fn policy_for_target(&self, target_alias: &str) -> anyhow::Result<Arc<SecurityPolicy>> {
        let Some(config) = self.root_config.as_ref() else {
            return Ok(Arc::clone(&self.security));
        };
        let mut target_policy = SecurityPolicy::for_agent(config, target_alias).map_err(|e| {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "target_agent": target_alias,
                        "error": format!("{}", e),
                    })),
                "delegate: could not resolve target's security policy"
            );
            anyhow::Error::msg(format!(
                "could not resolve security policy for delegate target {target_alias:?}: {e}"
            ))
        })?;
        if !self.security.delegation_policy.permits() {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "target_agent": target_alias,
                        "caller_risk_profile": self.security.risk_profile_name,
                    })),
                "delegate refused: caller delegation_policy forbids delegation"
            );
            return Err(anyhow::Error::msg(format!(
                "delegation is forbidden by the caller's delegation_policy; set \
                 [risk_profiles.{}].delegation_policy mode = \"allow\"",
                self.security.risk_profile_name
            )));
        }
        if let Err(reason) = self.cross_profile_decision(config, target_alias) {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "target_agent": target_alias,
                        "caller_risk_profile": self.security.risk_profile_name,
                        "target_risk_profile": target_policy.risk_profile_name,
                    })),
                "delegate refused: cross-profile narrowing gate"
            );
            return Err(anyhow::Error::msg(reason));
        }
        target_policy.tracker = self.security.tracker.clone();
        Ok(Arc::new(target_policy))
    }

    /// Per-target reachability decision for the narrowing-only delegation
    /// gate (see [`Self::policy_for_target`]). Returns `Ok(())` when the
    /// caller may delegate to `target_alias`, else `Err(reason)` with an
    /// operator-facing explanation. The caller's global delegation on/off
    /// switch (`delegation_policy.permits`) is enforced by the callers; this
    /// decides only the per-target narrowing / agentic rules with no side
    /// effects, so it is shared between enforcement
    /// ([`Self::policy_for_target`]) and roster advertisement
    /// ([`DelegateTool::parameters_schema`]).
    fn cross_profile_decision(&self, config: &Config, target_alias: &str) -> Result<(), String> {
        let Some(target_agent) = config.agents.get(target_alias) else {
            return Err(format!("unknown delegate target {target_alias:?}"));
        };
        let target_profile = target_agent.risk_profile.trim();
        // Same profile → identical grants; unchanged behavior (the agentic
        // loop's reuse of the caller's tool registry is safe when the
        // policies match).
        if self.security.risk_profile_name == target_profile {
            return Ok(());
        }
        // Cross-profile: compare profile GRANTS (not the per-agent resolved
        // policies — see `policy_for_target` doc). Without a resolvable caller
        // alias, fall back to the conservative same-profile requirement.
        let same_profile_required = || {
            format!(
                "delegate target {target_alias:?} uses risk profile {target_profile:?}, but \
                 delegation requires the same risk profile as the caller ({:?}), or a target \
                 whose profile is no broader",
                self.security.risk_profile_name
            )
        };
        if self.caller_alias.is_empty() {
            return Err(same_profile_required());
        }
        let (Some(caller_grants), Some(target_grants)) = (
            Self::profile_grants(config, &self.caller_alias),
            Self::profile_grants(config, target_alias),
        ) else {
            return Err(same_profile_required());
        };
        // Narrowing gate: the target's grants must be no broader than the
        // caller's. A BROADER target is a privilege escalation → refuse.
        //
        // NOTE — one filesystem dimension is deliberately NOT compared here:
        // each agent's OWN `workspace_dir` location/breadth (an operator-set
        // `agents.<a>.workspace.path`). `profile_grants` neutralizes it (both
        // agents share the sentinel base), because comparing it would refuse
        // ALL cross-profile delegation (sibling agents always have distinct,
        // non-nested own workspaces). A target with a custom `workspace.path`
        // onto a large/foreign subtree is therefore NOT caught by this gate.
        // It is inert on the only path the gate allows today (non-agentic,
        // toolless — no filesystem tool runs); it MUST be addressed before
        // cross-profile AGENTIC delegation (where the target would run FS
        // tools jailed to that workspace) is enabled. Tracked as a follow-up,
        // alongside the out-of-scope 0-sentinel shell_timeout/max_cost item.
        if let Err(escalation) = target_grants.ensure_no_escalation_beyond(&caller_grants) {
            return Err(format!(
                "delegate target {target_alias:?} (risk profile {target_profile:?}) would \
                 escalate beyond the caller's profile ({:?}): {escalation}. Delegation is \
                 narrowing-only — the target's permissions must be no broader than the caller's.",
                self.security.risk_profile_name
            ));
        }
        // `ensure_no_escalation_beyond` does not cover tool authorization;
        // close the `allowed_tools` / `excluded_tools` dimension explicitly.
        if let Err(tool) = Self::target_tools_within_caller(&caller_grants, &target_grants) {
            return Err(format!(
                "delegate target {target_alias:?} would authorize {tool}, which the caller's \
                 profile ({:?}) does not permit — delegation is narrowing-only.",
                self.security.risk_profile_name
            ));
        }
        // Nor does it cover the approval (auto_approve / always_ask) or
        // sandbox dimensions; a target that relaxes any of these runs broader
        // than the caller even when every grant above is narrower.
        if let Err(reason) =
            Self::target_approval_sandbox_within_caller(&caller_grants, &target_grants)
        {
            return Err(format!(
                "delegate target {target_alias:?} {reason}, which the caller's profile ({:?}) \
                 enforces — delegation is narrowing-only.",
                self.security.risk_profile_name
            ));
        }
        // Agentic cross-profile guard: the in-process agentic delegate loop
        // reuses the caller's `parent_tools` (bound to the caller's policy),
        // so a different-profile agentic target would run tools under the
        // CALLER's policy — an escalation. Non-agentic targets have no tool
        // registry and are safe.
        let target_agentic = config
            .runtime_profiles
            .get(target_agent.runtime_profile.trim())
            .map(|rp| rp.agentic)
            .unwrap_or(false);
        if target_agentic {
            return Err(format!(
                "delegate target {target_alias:?} is agentic and on a different risk profile \
                 ({target_profile:?}) than the caller ({:?}); cross-profile AGENTIC delegation \
                 is not supported on the in-process delegate path (the agentic loop reuses the \
                 caller's tool registry, bound to the caller's policy). Use spawn_subagent for a \
                 narrowed agentic run, or place the target on the caller's risk profile.",
                self.security.risk_profile_name
            ));
        }
        Ok(())
    }

    /// Build the capability-grants policy for `alias` used by the narrowing
    /// comparison in [`Self::cross_profile_decision`].
    ///
    /// Resolves the agent's risk + runtime profiles against a SHARED neutral
    /// base workspace ([`GRANT_CMP_WORKSPACE`]) via
    /// [`SecurityPolicy::from_profiles`]. Anchoring BOTH agents on the same
    /// base means every risk-profile `allowed_roots` entry — workspace-relative
    /// OR absolute — resolves to identical text for both, so a per-agent
    /// workspace jail (each agent's distinct real `workspace_dir`) never reads
    /// as a false escalation. (Anchoring on each agent's own `for_agent`
    /// workspace instead would sentinel-rewrite a shared absolute root for only
    /// the agent whose `workspace.path` happens to nest it, false-refusing
    /// provably-identical grants.)
    ///
    /// On top of that base it RE-APPLIES exactly the cross-agent filesystem
    /// tiers that `from_profiles` drops but the dispatched
    /// [`SecurityPolicy::for_agent`] policy carries, so the gate actually
    /// compares them (the gap this hardening closes): `workspace.access`
    /// sibling grants (per-config absolute paths, so genuine cross-agent
    /// broadening is preserved) and the `unrestricted_filesystem` escape hatch
    /// (which clears `workspace_only`). Persona-bundle equipment is merged
    /// against the shared base too, so same-persona agents still align while a
    /// persona that grants extra absolute roots/commands is compared. The
    /// per-agent own `workspace_dir` itself is deliberately NOT compared — see
    /// [`Self::cross_profile_decision`].
    ///
    /// `None` when the agent or its risk profile cannot be resolved (the
    /// caller then falls back to the conservative same-profile requirement).
    fn profile_grants(config: &Config, alias: &str) -> Option<SecurityPolicy> {
        use zeroclaw_config::multi_agent::AccessMode;
        let agent = config.agents.get(alias)?;
        let risk = config.risk_profiles.get(agent.risk_profile.trim())?;
        let runtime = config.runtime_profiles.get(agent.runtime_profile.trim());
        let mut policy =
            SecurityPolicy::from_profiles(risk, runtime, Path::new(GRANT_CMP_WORKSPACE));
        policy.risk_profile_name = agent.risk_profile.trim().to_string();
        // Mirror `for_agent`'s cross-agent tier resolution exactly (policy.rs),
        // minus the per-agent workspace anchoring we deliberately neutralize.
        for (sibling, mode) in &agent.workspace.access {
            let sibling_dir = config.agent_workspace_dir(sibling.as_str());
            match mode {
                AccessMode::Read => policy.allowed_roots_read_only.push(sibling_dir),
                AccessMode::Write => policy.allowed_roots_write_only.push(sibling_dir),
                AccessMode::ReadWrite => policy.allowed_roots.push(sibling_dir),
            }
        }
        if agent.workspace.unrestricted_filesystem {
            policy.workspace_only = false;
        }
        policy.merge_persona_bundle_equipment(config, alias);
        Some(policy)
    }

    /// Returns `Err(descriptor)` if the target's EFFECTIVE tool authorization
    /// admits a tool the caller's does not — i.e. the target's net tool set is
    /// not a subset of the caller's. Covers BOTH the `allowed_tools` allowlist
    /// (every arm, including the `(Some, None)` case where the target carries
    /// no allowlist while the caller restricts — an unrestricted target) AND
    /// `excluded_tools` (a smaller target denylist re-authorizes a tool the
    /// caller denies). [`SecurityPolicy::is_tool_allowed`] folds the
    /// allow+exclude pair into one predicate, so each arm reduces to "every
    /// tool the target can actually use must also be caller-usable".
    ///
    /// `descriptor` is a human-readable clause for the rejection message
    /// (`tool "shell"`, or `every tool …` for the unrestricted-target case).
    fn target_tools_within_caller(
        caller: &SecurityPolicy,
        target: &SecurityPolicy,
    ) -> Result<(), String> {
        match (&caller.allowed_tools, &target.allowed_tools) {
            // Caller restricts to an allowlist but the target carries none →
            // the target may run any tool the caller's allowlist omits.
            (Some(_), None) => Err(
                "every tool (the target has no allowed_tools allowlist while the caller restricts to one)"
                    .to_string(),
            ),
            // Target carries an explicit allowlist (caller restricted or not):
            // every tool the target can actually use (allowed AND not
            // self-excluded) must also be usable by the caller.
            (_, Some(target_allow)) => {
                for name in target_allow {
                    if target.is_tool_allowed(name) && !caller.is_tool_allowed(name) {
                        return Err(format!("tool {name:?}"));
                    }
                }
                Ok(())
            }
            // Neither restricts via an allowlist: only the denylists differ.
            // Every tool the caller denies must stay denied by the target,
            // else the target re-authorizes it.
            (None, None) => {
                if let Some(caller_excluded) = &caller.excluded_tools {
                    for name in caller_excluded {
                        if target.is_tool_allowed(name) {
                            return Err(format!("tool {name:?} (which the caller excludes)"));
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Returns `Err(reason)` when the target relaxes an APPROVAL or SANDBOX
    /// constraint the caller enforces — dimensions `ensure_no_escalation_beyond`
    /// and `target_tools_within_caller` do not cover. Each is "target at least
    /// as restrictive as the caller":
    /// * `auto_approve`: the target must not auto-approve (skip the approval
    ///   prompt for) any tool — or the `*` blanket — the caller does not
    ///   auto-approve. (`auto_approve ⊆ caller.auto_approve`.)
    /// * `always_ask`: the target must not DROP an `always_ask` the caller
    ///   requires (the `*` blanket included). (`caller.always_ask ⊆ target`.)
    /// * sandbox: when the caller runs EFFECTIVELY sandboxed (see
    ///   [`Self::effectively_sandboxed`] — which mirrors runtime enforcement,
    ///   so the common backend-set / `sandbox_enabled`-unset default counts),
    ///   the target must too, on the SAME resolved backend. (When the caller is
    ///   not effectively sandboxed the dimension is unconstrained — avoids
    ///   false-refusing the symmetric default configuration.) `firejail_args` is
    ///   deliberately NOT compared: the runtime hard-codes the firejail flag set
    ///   and never forwards `firejail_args` (see `security::firejail`), so it is
    ///   runtime-inert today and comparing it would only false-refuse
    ///   runtime-identical delegations — see the inline note for the comparison
    ///   to add once the runtime wires it through.
    ///
    /// `reason` is a human-readable verb-phrase for the rejection message.
    fn target_approval_sandbox_within_caller(
        caller: &SecurityPolicy,
        target: &SecurityPolicy,
    ) -> Result<(), String> {
        // auto_approve: target ⊆ caller (a target `*` requires a caller `*`).
        let caller_auto_all = caller.auto_approve.iter().any(|t| t == "*");
        for tool in &target.auto_approve {
            if !caller_auto_all && !caller.auto_approve.iter().any(|t| t == tool) {
                return Err(format!(
                    "auto-approves {tool:?}, bypassing an approval prompt the caller requires"
                ));
            }
        }
        // always_ask: caller ⊆ target (the target may not drop a requirement).
        let target_ask_all = target.always_ask.iter().any(|t| t == "*");
        for tool in &caller.always_ask {
            if !target_ask_all && !target.always_ask.iter().any(|t| t == tool) {
                return Err(format!(
                    "drops the always-ask approval the caller requires for {tool:?}"
                ));
            }
        }
        // sandbox: compare EFFECTIVE sandbox state, mirroring runtime
        // enforcement (zeroclaw-config `RiskProfileConfig::sandbox_config` +
        // `security::detect::create_sandbox`): a policy runs sandboxed iff its
        // resolved backend is not `none` AND `sandbox_enabled != Some(false)`
        // (an unset backend resolves to `auto`; an unset `sandbox_enabled`
        // leaves the sandbox ACTIVE). The earlier `sandbox_enabled == Some(true)`
        // guard missed that active-by-default regime, letting a strictly
        // unsandboxed target slip past.
        if Self::effectively_sandboxed(caller) {
            if !Self::effectively_sandboxed(target) {
                return Err(
                    "runs unsandboxed where the caller runs sandboxed (sandbox_enabled / sandbox_backend)"
                        .to_string(),
                );
            }
            // Both run sandboxed. Backends are not orderable, so require the
            // SAME resolved backend (a different backend is not provably
            // no-weaker).
            let caller_backend = Self::resolved_sandbox_backend(caller);
            let target_backend = Self::resolved_sandbox_backend(target);
            if caller_backend != target_backend {
                return Err(format!(
                    "uses sandbox backend {target_backend:?} where the caller uses {caller_backend:?} (a different backend is not provably narrower)"
                ));
            }
            // NOTE: `firejail_args` is intentionally NOT compared. The runtime
            // `FirejailSandbox::wrap_command` hard-codes its flag set and never
            // forwards policy `firejail_args` to the firejail invocation, so the
            // field is runtime-INERT: two profiles differing only in
            // `firejail_args` produce a byte-identical sandbox, and comparing it
            // would false-refuse runtime-identical delegations. When the runtime
            // starts forwarding `firejail_args`, add an EQUALITY check here — the
            // arg space is non-monotone (e.g. `--noprofile` loosens confinement
            // while keeping every caller flag), so subset containment is unsound.
        }
        Ok(())
    }

    /// Resolve a policy's sandbox backend to the normalized token the runtime
    /// uses, mirroring `RiskProfileConfig::sandbox_config` (trim / lowercase /
    /// unset → `auto`) + `parse_sandbox_backend`. Kept as a string token so
    /// this crate need not reach into the (private) parser; the trailing
    /// `_ => "auto"` arm matches the parser's `_ => SandboxBackend::default()`
    /// so unknown or newly-added names behave identically on both sides.
    fn resolved_sandbox_backend(policy: &SecurityPolicy) -> &'static str {
        match policy
            .sandbox_backend
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None | Some("auto") => "auto",
            Some("landlock") => "landlock",
            Some("firejail") => "firejail",
            Some("bubblewrap") => "bubblewrap",
            Some("docker") => "docker",
            Some("sandbox-exec" | "sandboxexec" | "seatbelt") => "sandbox-exec",
            Some("none") => "none",
            Some(_) => "auto",
        }
    }

    /// True when `policy` runs under a real sandbox at enforcement time,
    /// mirroring `security::detect::create_sandbox`: a `NoopSandbox` results
    /// iff the resolved backend is `none` OR `sandbox_enabled == Some(false)`.
    fn effectively_sandboxed(policy: &SecurityPolicy) -> bool {
        Self::resolved_sandbox_backend(policy) != "none" && policy.sandbox_enabled != Some(false)
    }

    /// Resolve `model_provider` ("type.alias") → (provider_type, credential, model, temperature).
    fn resolve_brain(&self, model_provider: &str) -> (String, Option<String>, String, Option<f64>) {
        if let Some((type_key, alias_key)) = model_provider.split_once('.')
            && let Some(alias_map) = self.providers_models.get(type_key)
            && let Some(cfg) = alias_map.get(alias_key)
        {
            return (
                type_key.to_string(),
                cfg.api_key
                    .clone()
                    .or_else(|| self.global_credential.clone()),
                cfg.model.clone().unwrap_or_default(),
                cfg.temperature,
            );
        }
        let type_key = model_provider
            .split_once('.')
            .map_or(model_provider, |(t, _)| t);
        (
            type_key.to_string(),
            self.global_credential.clone(),
            String::new(),
            None,
        )
    }

    /// Resolve `ModelProviderRuntimeOptions` for a delegate's OWN
    /// `model_provider` alias (`"<family>.<alias>"`). Re-resolving per delegate
    /// alias makes each sub-agent target its own family endpoint (the explicit
    /// alias `uri`, else the family default via `resolved_endpoint_uri`) instead
    /// of reusing the single inherited `provider_runtime_options`. Falls back to
    /// the inherited options when the alias is bare (no `family.alias`) or no
    /// `root_config` is attached (legacy unit-test constructors).
    fn resolve_delegate_provider_options(
        &self,
        model_provider: &str,
    ) -> zeroclaw_providers::ModelProviderRuntimeOptions {
        match (model_provider.split_once('.'), self.root_config.as_deref()) {
            (Some((family, alias)), Some(config)) => {
                zeroclaw_providers::provider_runtime_options_for_alias(config, family, alias)
            }
            _ => self.provider_runtime_options.clone(),
        }
    }

    /// Resolve max delegation depth from the named runtime profile (default: 3).
    fn resolve_max_depth(&self, runtime_profile: &str) -> u32 {
        if runtime_profile.is_empty() {
            return 3;
        }
        self.runtime_profiles
            .get(runtime_profile)
            .map(|p| p.max_delegation_depth)
            .filter(|&d| d > 0)
            .unwrap_or(3)
    }

    /// Resolve per-call delegation timeout from the named runtime profile.
    fn resolve_delegation_timeout(&self, runtime_profile: &str) -> Option<u64> {
        if runtime_profile.is_empty() {
            return None;
        }
        self.runtime_profiles
            .get(runtime_profile)
            .and_then(|p| p.delegation_timeout_secs)
    }

    /// Resolve agentic run timeout from the named runtime profile.
    fn resolve_agentic_timeout_secs(&self, runtime_profile: &str) -> Option<u64> {
        if runtime_profile.is_empty() {
            return None;
        }
        self.runtime_profiles
            .get(runtime_profile)
            .and_then(|p| p.agentic_timeout_secs)
    }

    /// Resolve agentic mode flag from the named runtime profile (default: false).
    ///
    /// Trims the reference before lookup so this dispatch-side resolution can
    /// never disagree with [`Self::cross_profile_decision`]'s gate, which reads
    /// `agentic` via a TRIMMED `runtime_profiles.get(...)`. A whitespace-padded
    /// `runtime_profile` must not make the gate read non-agentic (and allow a
    /// cross-profile delegate) while dispatch reads agentic (and runs the
    /// reused-tool-registry loop), or vice versa.
    fn resolve_agentic(&self, runtime_profile: &str) -> bool {
        let runtime_profile = runtime_profile.trim();
        if runtime_profile.is_empty() {
            return false;
        }
        self.runtime_profiles
            .get(runtime_profile)
            .map(|p| p.agentic)
            .unwrap_or(false)
    }

    /// Resolve max tool iterations from the named runtime profile (default: 10).
    fn resolve_max_iterations(&self, runtime_profile: &str) -> usize {
        if runtime_profile.is_empty() {
            return 10;
        }
        self.runtime_profiles
            .get(runtime_profile)
            .map(|p| p.max_tool_iterations)
            .filter(|&i| i > 0)
            .unwrap_or(10)
    }

    /// Resolve allowed tools list from the named risk profile (authorization).
    fn resolve_allowed_tools(&self, risk_profile: &str) -> Vec<String> {
        if risk_profile.is_empty() {
            return Vec::new();
        }
        self.risk_profiles
            .get(risk_profile)
            .map(|p| p.allowed_tools.clone())
            .unwrap_or_default()
    }

    /// Resolve every configured skill bundle alias to its directory.
    /// Empty list / no matches → caller falls back to the workspace default.
    fn resolve_skill_bundle_dirs(&self, bundle_aliases: &[String]) -> Vec<String> {
        bundle_aliases
            .iter()
            .filter(|a| !a.is_empty())
            .filter_map(|a| self.skill_bundles.get(a).and_then(|b| b.directory.clone()))
            .collect()
    }

    /// Directory where background delegate results are stored.
    fn results_dir(&self) -> PathBuf {
        self.workspace_dir.join("delegate_results")
    }

    /// Validate that a user-provided task_id is a valid UUID to prevent
    /// path traversal attacks (e.g. `../../etc/passwd`).
    fn validate_task_id(task_id: &str) -> Result<(), String> {
        if uuid::Uuid::parse_str(task_id).is_err() {
            return Err(format!("Invalid task_id '{task_id}': must be a valid UUID"));
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for DelegateTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn description(&self) -> &str {
        "Delegate a subtask to a specialized agent. Use when: a task benefits from a different model \
         (e.g. fast summarization, deep reasoning, code generation). The sub-agent runs a single \
         prompt by default; with agentic=true it can iterate with a filtered tool-call loop. \
         Supports background execution (returns a task_id immediately) and parallel execution \
         (runs multiple agents concurrently). Use action='check_result' with a task_id to \
         retrieve background results."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let delegation_permitted = self.security.delegation_policy.permits();
        let caller_profile = self.security.risk_profile_name.as_str();
        // Advertise only agents the caller can actually reach: delegation must
        // be permitted, the delegator never lists itself, and the target must
        // pass the narrowing-only gate — same-profile, or a different profile
        // that is no broader than the caller's and non-agentic (mirrors
        // `cross_profile_decision`). Without `root_config` (legacy unit-test
        // constructors) fall back to the same-profile filter.
        let mut agent_names: Vec<&str> = self
            .agents
            .iter()
            .filter(|_| delegation_permitted)
            .filter(|(name, _)| name.as_str() != self.caller_alias.as_str())
            .filter(|(name, cfg)| match self.root_config.as_ref() {
                Some(config) => self.cross_profile_decision(config, name).is_ok(),
                None => cfg.risk_profile.trim() == caller_profile,
            })
            .map(|(name, _)| name.as_str())
            .collect();
        agent_names.sort_unstable();
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["delegate", "check_result", "list_results", "cancel_task"],
                    "description": "Action to perform. Default: 'delegate'. Use 'check_result' to \
                                    retrieve a background task result, 'list_results' to list all \
                                    background tasks, 'cancel_task' to cancel a running background task.",
                    "default": "delegate"
                },
                "agent": {
                    "type": "string",
                    "minLength": 1,
                    "description": format!(
                        "Name of the agent to delegate to. Available: {}",
                        if agent_names.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            agent_names.join(", ")
                        }
                    )
                },
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The task/prompt to send to the sub-agent"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to prepend (e.g. relevant code, prior findings)"
                },
                "background": {
                    "type": "boolean",
                    "description": "When true, the sub-agent runs in a background tokio task and \
                                    returns a task_id immediately. Results are stored to \
                                    workspace/delegate_results/{task_id}.json.",
                    "default": false
                },
                "parallel": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Array of agent names to run concurrently with the same prompt. \
                                    Returns all results when all agents complete. Cannot be combined \
                                    with 'background'."
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID for check_result/cancel_task actions (returned by \
                                    background delegation)."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("delegate");

        match action {
            "check_result" => return self.handle_check_result(&args).await,
            "list_results" => return self.handle_list_results().await,
            "cancel_task" => return self.handle_cancel_task(&args).await,
            "delegate" => {} // fall through to delegation logic
            other => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Unknown action '{other}'. Use delegate/check_result/list_results/cancel_task."
                    )),
                });
            }
        }

        // --- Parallel mode ---
        if let Some(parallel_agents) = args.get("parallel").and_then(|v| v.as_array()) {
            return self.execute_parallel(parallel_agents, &args).await;
        }

        // --- Single-agent delegation (synchronous or background) ---
        let agent_name = args
            .get("agent")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"param": "agent"})),
                    "tool argument validation failed"
                );

                anyhow::Error::msg("Missing 'agent' parameter")
            })?;

        if agent_name.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'agent' parameter must not be empty".into()),
            });
        }

        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"param": "prompt"})),
                    "tool argument validation failed"
                );

                anyhow::Error::msg("Missing 'prompt' parameter")
            })?;

        if prompt.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'prompt' parameter must not be empty".into()),
            });
        }

        let background = args
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if background {
            return self.execute_background(agent_name, prompt, &args).await;
        }

        // --- Synchronous delegation (original path) ---
        self.execute_sync(agent_name, prompt, &args).await
    }
}

impl DelegateTool {
    /// Original synchronous delegation path (extracted for reuse).
    async fn execute_sync(
        &self,
        agent_name: &str,
        prompt: &str,
        args: &serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        let context = args
            .get("context")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or("");

        // Look up agent config
        let agent_config = match self.agents.get(agent_name) {
            Some(cfg) => cfg,
            None => {
                let available: Vec<&str> =
                    self.agents.keys().map(|s: &String| s.as_str()).collect();
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Unknown agent '{agent_name}'. Available agents: {}",
                        if available.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            available.join(", ")
                        }
                    )),
                });
            }
        };

        // Resolve profile references
        let max_depth = self.resolve_max_depth(&agent_config.runtime_profile);
        let (provider_type, credential, model, temperature) =
            self.resolve_brain(&agent_config.model_provider);
        let agentic = self.resolve_agentic(&agent_config.runtime_profile);

        // Check recursion depth (immutable — set at construction, incremented for sub-agents)
        if self.depth >= max_depth {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Delegation depth limit reached ({depth}/{max}). \
                     Cannot delegate further to prevent infinite loops.",
                    depth = self.depth,
                    max = max_depth
                )),
            });
        }

        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "delegate")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        if let Err(e) = self.policy_for_target(agent_name) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("{e:#}")),
            });
        }

        // Create model_provider for this agent. Re-resolve runtime options for
        // THIS delegate's own alias so an ollama (or any local) delegate posts
        // to its own family endpoint instead of inheriting the shared
        // `self.provider_runtime_options` (see `resolve_delegate_provider_options`).
        let delegate_provider_options =
            self.resolve_delegate_provider_options(&agent_config.model_provider);
        let model_provider: Box<dyn ModelProvider> =
            match zeroclaw_providers::create_model_provider_with_options(
                &provider_type,
                credential.as_deref(),
                &delegate_provider_options,
            ) {
                Ok(p) => p,
                Err(e) => {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!(
                            "Failed to create model_provider '{provider_type}' for agent '{agent_name}': {e}"
                        )),
                    });
                }
            };

        // Build the message
        let full_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("[Context]\n{context}\n\n[Task]\n{prompt}")
        };

        // Agentic mode: run full tool-call loop with allowlisted tools.
        if agentic {
            return self
                .execute_agentic(
                    agent_name,
                    agent_config,
                    &provider_type,
                    &model,
                    &*model_provider,
                    &full_prompt,
                    temperature,
                )
                .await;
        }

        // Build enriched system prompt for non-agentic sub-agent.
        let enriched_system_prompt = self.build_enriched_system_prompt(
            agent_name,
            agent_config,
            &model,
            &[],
            &self.workspace_dir,
            false,
        );
        let system_prompt_ref = enriched_system_prompt.as_deref();

        // Assemble a (system?, user) message list so we can call chat() rather
        // than chat_with_system(): chat() returns a ChatResponse whose `usage`
        // we surface as a per-delegate LlmResponse observer event below.
        // chat_with_system() returns only a String and discards token counts —
        // non-agentic delegates bypass the agent loop, so without this they emit
        // no inner-call telemetry at all (agentic delegates emit via the loop).
        let mut inner_messages: Vec<ChatMessage> = Vec::with_capacity(2);
        if let Some(system) = system_prompt_ref {
            inner_messages.push(ChatMessage::system(system));
        }
        inner_messages.push(ChatMessage::user(full_prompt.clone()));
        let chat_request = ChatRequest {
            messages: &inner_messages,
            tools: None,
            thinking: None,
        };

        // Wrap the model_provider call in a timeout to prevent indefinite blocking
        let timeout_secs = self
            .resolve_delegation_timeout(&agent_config.runtime_profile)
            .unwrap_or(self.delegate_config.timeout_secs);
        let inner_started_at = std::time::Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            model_provider.chat(chat_request, &model, temperature),
        )
        .await;

        let result = match result {
            Ok(inner) => inner,
            Err(_elapsed) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Agent '{agent_name}' timed out after {timeout_secs}s"
                    )),
                });
            }
        };

        match result {
            Ok(response) => {
                // Per-delegate inner-call telemetry: emit an LlmResponse so
                // zc-delegate-stats can attribute tokens + latency to this
                // non-agentic delegate. provider/model are the configured ones
                // (no cross-provider fallback on the delegate path in this
                // build — see the deferred per-agent fallback work).
                if let Some(observer) = &self.observer {
                    let (input_tokens, output_tokens) = response
                        .usage
                        .as_ref()
                        .map(|u| (u.input_tokens, u.output_tokens))
                        .unwrap_or((None, None));
                    observer.record_event(&ObserverEvent::LlmResponse {
                        model_provider: provider_type.clone(),
                        model: model.clone(),
                        duration: inner_started_at.elapsed(),
                        success: true,
                        error_message: None,
                        input_tokens,
                        output_tokens,
                    });
                }

                let mut rendered = response.text.unwrap_or_default();
                if rendered.trim().is_empty() {
                    rendered = "[Empty response]".to_string();
                }

                Ok(ToolResult {
                    success: true,
                    output: format!("[Agent '{agent_name}' ({provider_type}/{model})]\n{rendered}",),
                    error: None,
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}",)),
            }),
        }
    }
}

impl DelegateTool {
    // ── Background Execution ────────────────────────────────────────

    /// Spawn a sub-agent in a background tokio task. Returns a task_id immediately.
    /// The result is persisted to `workspace/delegate_results/{task_id}.json`.
    async fn execute_background(
        &self,
        agent_name: &str,
        prompt: &str,
        args: &serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        // Validate agent exists and check depth/security before spawning
        let agent_config = match self.agents.get(agent_name) {
            Some(cfg) => cfg.clone(),
            None => {
                let available: Vec<&str> =
                    self.agents.keys().map(|s: &String| s.as_str()).collect();
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Unknown agent '{agent_name}'. Available agents: {}",
                        if available.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            available.join(", ")
                        }
                    )),
                });
            }
        };

        let max_depth = self.resolve_max_depth(&agent_config.runtime_profile);
        if self.depth >= max_depth {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Delegation depth limit reached ({depth}/{max}).",
                    depth = self.depth,
                    max = max_depth
                )),
            });
        }

        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "delegate")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let target_policy = match self.policy_for_target(agent_name) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("{e:#}")),
                });
            }
        };

        let task_id = uuid::Uuid::new_v4().to_string();
        let results_dir = self.results_dir();
        tokio::fs::create_dir_all(&results_dir).await?;

        let context = args
            .get("context")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or("");
        let full_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("[Context]\n{context}\n\n[Task]\n{prompt}")
        };

        let started_at = chrono::Utc::now().to_rfc3339();
        let agent_name_owned = agent_name.to_string();

        // Write initial "running" status
        let initial_result = BackgroundDelegateResult {
            task_id: task_id.clone(),
            agent: agent_name_owned.clone(),
            status: BackgroundTaskStatus::Running,
            output: None,
            error: None,
            started_at: started_at.clone(),
            finished_at: None,
        };
        let result_path = results_dir.join(format!("{task_id}.json"));
        let json_bytes = serde_json::to_vec_pretty(&initial_result)?;
        tokio::fs::write(&result_path, &json_bytes).await?;

        let agents = Arc::clone(&self.agents);
        let security = target_policy;
        let global_credential = self.global_credential.clone();
        let provider_runtime_options = self.provider_runtime_options.clone();
        let depth = self.depth;
        let parent_tools = Arc::clone(&self.parent_tools);
        let multimodal_config = self.multimodal_config.clone();
        let delegate_config = self.delegate_config.clone();
        let workspace_dir = self.workspace_dir.clone();
        let child_token = self.cancellation_token.child_token();
        let task_id_clone = task_id.clone();
        let providers_models = Arc::clone(&self.providers_models);
        let risk_profiles = Arc::clone(&self.risk_profiles);
        let runtime_profiles = Arc::clone(&self.runtime_profiles);
        let skill_bundles = Arc::clone(&self.skill_bundles);
        let root_config = self.root_config.clone();
        let observer = self.observer.clone();
        let caller_alias = self.caller_alias.clone();
        // Capture the parent loop's session-key task-local so the
        // detached background task scopes its tool calls under the
        // same key — channel tools (sessions_send, etc.) need the
        // session key in scope to attribute correctly. Without this
        // wrap, the spawned task would lose the parent's task-local
        // and channel-scoped tool calls would land unattributed.
        let parent_session_key = current_tool_loop_session_key();
        let __zc_delegate_alias = agent_name_owned.clone();

        zeroclaw_spawn::spawn!(
            scope_delegate_session_key(parent_session_key, async move {
                let inner = DelegateTool {
                    agents,
                    security,
                    global_credential,
                    provider_runtime_options,
                    depth,
                    parent_tools,
                    multimodal_config,
                    delegate_config,
                    workspace_dir: workspace_dir.clone(),
                    cancellation_token: child_token.clone(),
                    memory: None,
                    providers_models,
                    risk_profiles,
                    runtime_profiles,
                    skill_bundles,
                    root_config,
                    observer,
                    caller_alias,
                };

                let args_inner = json!({
                    "agent": agent_name_owned,
                    "prompt": full_prompt,
                });

                // Race the delegation against cancellation
                let outcome = tokio::select! {
                    () = child_token.cancelled() => {
                        Err("Cancelled by parent session".to_string())
                    }
                    result = Box::pin(inner.execute_sync(&agent_name_owned, &full_prompt, &args_inner)) => {
                        match result {
                            Ok(tool_result) => {
                                if tool_result.success {
                                    Ok(tool_result.output)
                                } else {
                                    Err(tool_result.error.unwrap_or_else(|| "Unknown error".into()))
                                }
                            }
                            Err(e) => Err(e.to_string()),
                        }
                    }
                };

                let finished_at = chrono::Utc::now().to_rfc3339();
                let final_result = match outcome {
                    Ok(output) => BackgroundDelegateResult {
                        task_id: task_id_clone.clone(),
                        agent: agent_name_owned,
                        status: BackgroundTaskStatus::Completed,
                        output: Some(output),
                        error: None,
                        started_at,
                        finished_at: Some(finished_at),
                    },
                    Err(err) => {
                        let status = if err.contains("Cancelled") {
                            BackgroundTaskStatus::Cancelled
                        } else {
                            BackgroundTaskStatus::Failed
                        };
                        BackgroundDelegateResult {
                            task_id: task_id_clone.clone(),
                            agent: agent_name_owned,
                            status,
                            output: None,
                            error: Some(err),
                            started_at,
                            finished_at: Some(finished_at),
                        }
                    }
                };

                let result_path = results_dir.join(format!("{}.json", task_id_clone));
                if let Ok(bytes) = serde_json::to_vec_pretty(&final_result) {
                    let _ = tokio::fs::write(&result_path, &bytes).await;
                }
            })
            .instrument(::zeroclaw_log::attribution_span!(
                &crate::agent::AgentAttribution(__zc_delegate_alias.as_str())
            ))
        );

        Ok(ToolResult {
            success: true,
            output: format!(
                "Background task started for agent '{agent_name}'.\n\
                 task_id: {task_id}\n\
                 Use action='check_result' with task_id='{task_id}' to retrieve the result."
            ),
            error: None,
        })
    }

    // ── Parallel Execution ──────────────────────────────────────────

    /// Run multiple agents concurrently with the same prompt.
    async fn execute_parallel(
        &self,
        parallel_agents: &[serde_json::Value],
        args: &serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"param": "prompt"})),
                    "tool argument validation failed"
                );

                anyhow::Error::msg("Missing 'prompt' parameter for parallel execution")
            })?;

        if prompt.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'prompt' parameter must not be empty".into()),
            });
        }

        let agent_names: Vec<String> = parallel_agents
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect();

        if agent_names.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'parallel' array must contain at least one agent name".into()),
            });
        }

        // Validate all agents exist before starting any
        for name in &agent_names {
            if !self.agents.contains_key(name) {
                let available: Vec<&str> =
                    self.agents.keys().map(|s: &String| s.as_str()).collect();
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Unknown agent '{name}' in parallel list. Available: {}",
                        if available.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            available.join(", ")
                        }
                    )),
                });
            }
        }

        let mut target_policies: HashMap<String, Arc<SecurityPolicy>> =
            HashMap::with_capacity(agent_names.len());
        for name in &agent_names {
            match self.policy_for_target(name) {
                Ok(p) => {
                    target_policies.insert(name.clone(), p);
                }
                Err(e) => {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("{e:#}")),
                    });
                }
            }
        }

        // Capture the current receipt scope so each spawned sub-agent task
        // re-enters it. Spawned tasks do not propagate task-locals, so
        // without this `execute_sync`'s `try_with` would resolve to `None`
        // inside the spawn and the parallel agents would run unsigned even
        // when the parent turn has receipts enabled. The collector is `Arc`'d
        // inside `ReceiptScope`, so all parallel agents push into the same
        // per-turn collector the orchestrator renders after the loop returns.
        let parent_receipt_scope = crate::agent::tool_receipts::TOOL_LOOP_RECEIPT_CONTEXT
            .try_with(Clone::clone)
            .ok()
            .flatten();
        let parent_session_key = current_tool_loop_session_key();

        // Spawn all agents concurrently
        let mut handles = Vec::with_capacity(agent_names.len());
        for agent_name in &agent_names {
            let agents = Arc::clone(&self.agents);
            let security = target_policies
                .get(agent_name)
                .cloned()
                .unwrap_or_else(|| Arc::clone(&self.security));
            let global_credential = self.global_credential.clone();
            let provider_runtime_options = self.provider_runtime_options.clone();
            let depth = self.depth;
            let parent_tools = Arc::clone(&self.parent_tools);
            let multimodal_config = self.multimodal_config.clone();
            let delegate_config = self.delegate_config.clone();
            let workspace_dir = self.workspace_dir.clone();
            let cancellation_token = self.cancellation_token.child_token();
            let agent_name = agent_name.clone();
            let prompt = prompt.to_string();
            let args_clone = args.clone();
            let providers_models = Arc::clone(&self.providers_models);
            let risk_profiles = Arc::clone(&self.risk_profiles);
            let runtime_profiles = Arc::clone(&self.runtime_profiles);
            let skill_bundles = Arc::clone(&self.skill_bundles);
            let receipt_scope = parent_receipt_scope.clone();
            let root_config = self.root_config.clone();
            let observer = self.observer.clone();
            let caller_alias = self.caller_alias.clone();
            let session_key = parent_session_key.clone();
            let __zc_delegate_alias = agent_name.clone();

            handles.push(zeroclaw_spawn::spawn!(
                async move {
                    let inner = DelegateTool {
                        agents,
                        security,
                        global_credential,
                        provider_runtime_options,
                        depth,
                        parent_tools,
                        multimodal_config,
                        delegate_config,
                        workspace_dir,
                        cancellation_token,
                        memory: None,
                        providers_models,
                        risk_profiles,
                        runtime_profiles,
                        skill_bundles,
                        root_config,
                        observer,
                        caller_alias,
                    };
                    let agent_name_for_return = agent_name.clone();
                    let result = scope_delegate_session_key(session_key, async move {
                        crate::agent::tool_receipts::TOOL_LOOP_RECEIPT_CONTEXT
                            .scope(receipt_scope, async move {
                                Box::pin(inner.execute_sync(&agent_name, &prompt, &args_clone))
                                    .await
                            })
                            .await
                    })
                    .await;
                    (agent_name_for_return, result)
                }
                .instrument(::zeroclaw_log::attribution_span!(
                    &crate::agent::AgentAttribution(__zc_delegate_alias.as_str())
                ))
            ));
        }

        // Collect all results
        let mut outputs = Vec::with_capacity(handles.len());
        let mut all_success = true;

        for handle in handles {
            match handle.await {
                Ok((agent_name, Ok(tool_result))) => {
                    if !tool_result.success {
                        all_success = false;
                    }
                    outputs.push(format!(
                        "--- {agent_name} (success={}) ---\n{}{}",
                        tool_result.success,
                        tool_result.output,
                        tool_result
                            .error
                            .map(|e| format!("\nError: {e}"))
                            .unwrap_or_default()
                    ));
                }
                Ok((agent_name, Err(e))) => {
                    all_success = false;
                    outputs.push(format!("--- {agent_name} (success=false) ---\nError: {e}"));
                }
                Err(e) => {
                    all_success = false;
                    outputs.push(format!("--- [join error] ---\n{e}"));
                }
            }
        }

        Ok(ToolResult {
            success: all_success,
            output: format!(
                "[Parallel delegation: {} agents]\n\n{}",
                agent_names.len(),
                outputs.join("\n\n")
            ),
            error: if all_success {
                None
            } else {
                Some("One or more parallel agents failed".into())
            },
        })
    }

    // ── Result Retrieval ────────────────────────────────────────────

    /// Retrieve the result of a background delegate task by task_id.
    async fn handle_check_result(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let task_id = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"param": "task_id"})),
                    "tool argument validation failed"
                );

                anyhow::Error::msg("Missing 'task_id' parameter for check_result")
            })?;

        if let Err(e) = Self::validate_task_id(task_id) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e),
            });
        }

        let result_path = self.results_dir().join(format!("{task_id}.json"));
        if !result_path.exists() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("No result found for task_id '{task_id}'")),
            });
        }

        let content = tokio::fs::read_to_string(&result_path).await?;
        let result: BackgroundDelegateResult = serde_json::from_str(&content)?;

        Ok(ToolResult {
            success: result.status == BackgroundTaskStatus::Completed,
            output: serde_json::to_string_pretty(&result)?,
            error: if result.status == BackgroundTaskStatus::Completed {
                None
            } else {
                result.error
            },
        })
    }

    /// List all background delegate task results.
    async fn handle_list_results(&self) -> anyhow::Result<ToolResult> {
        let results_dir = self.results_dir();
        if !results_dir.exists() {
            return Ok(ToolResult {
                success: true,
                output: "No background delegate results found.".into(),
                error: None,
            });
        }

        let mut entries = tokio::fs::read_dir(&results_dir).await?;
        let mut results = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json")
                && let Ok(content) = tokio::fs::read_to_string(&path).await
                && let Ok(result) = serde_json::from_str::<BackgroundDelegateResult>(&content)
            {
                results.push(json!({
                    "task_id": result.task_id,
                    "agent": result.agent,
                    "status": result.status,
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                }));
            }
        }

        if results.is_empty() {
            return Ok(ToolResult {
                success: true,
                output: "No background delegate results found.".into(),
                error: None,
            });
        }

        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&results)?,
            error: None,
        })
    }

    /// Cancel a running background task by task_id.
    async fn handle_cancel_task(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let task_id = args
            .get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"param": "task_id"})),
                    "tool argument validation failed"
                );

                anyhow::Error::msg("Missing 'task_id' parameter for cancel_task")
            })?;

        if let Err(e) = Self::validate_task_id(task_id) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e),
            });
        }

        let result_path = self.results_dir().join(format!("{task_id}.json"));
        if !result_path.exists() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("No task found for task_id '{task_id}'")),
            });
        }

        // Read current status
        let content = tokio::fs::read_to_string(&result_path).await?;
        let mut result: BackgroundDelegateResult = serde_json::from_str(&content)?;

        if result.status != BackgroundTaskStatus::Running {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Task '{task_id}' is not running (status: {:?})",
                    result.status
                )),
            });
        }

        // Cancel via the parent token — this will cascade to all child tokens
        // Note: individual task cancellation uses the shared parent token, which
        // cancels all background tasks. For per-task cancellation, each background
        // task uses a child token, and the parent token cancels all.
        // We update the result file to reflect the cancellation request.
        result.status = BackgroundTaskStatus::Cancelled;
        result.error = Some("Cancelled by user request".into());
        result.finished_at = Some(chrono::Utc::now().to_rfc3339());
        let bytes = serde_json::to_vec_pretty(&result)?;
        tokio::fs::write(&result_path, &bytes).await?;

        Ok(ToolResult {
            success: true,
            output: format!("Task '{task_id}' cancellation requested."),
            error: None,
        })
    }

    /// Cancel all background tasks (cascade control).
    /// Call this when the parent session ends.
    pub fn cancel_all_background_tasks(&self) {
        self.cancellation_token.cancel();
    }

    /// Build an enriched system prompt for a sub-agent by composing structured
    /// operational sections (tools, skills, workspace, datetime, shell policy)
    /// with the per-agent identity files loaded from the target's own
    /// workspace dir (`<install>/agents/<alias>/workspace/AGENTS.md`,
    /// `SOUL.md`, `IDENTITY.md`, `USER.md`, `TOOLS.md`, `BOOTSTRAP.md`,
    /// `MEMORY.md`).
    fn build_enriched_system_prompt(
        &self,
        agent_alias: &str,
        agent_config: &AliasedAgentConfig,
        model_name: &str,
        sub_tools: &[Box<dyn Tool>],
        workspace_dir: &Path,
        sends_native_tool_specs: bool,
    ) -> Option<String> {
        // Resolve skill bundle directories. With one or more configured
        // bundles, load + concat skills from each. With none, fall back to
        // the workspace default.
        let bundle_dirs = self.resolve_skill_bundle_dirs(&agent_config.skill_bundles);
        let skills = if bundle_dirs.is_empty() {
            let default_dir = crate::skills::skills_dir(workspace_dir);
            crate::skills::load_skills_from_directory(&default_dir, false)
        } else {
            bundle_dirs
                .into_iter()
                .flat_map(|dir| {
                    crate::skills::load_skills_from_directory(&workspace_dir.join(dir), false)
                })
                .collect()
        };

        // Determine shell policy instructions when the `shell` tool is in the
        // effective tool list.
        let empty_tools: &[Box<dyn Tool>] = &[];
        let expose_text_tools =
            sends_native_tool_specs || !agent_config.resolved.strict_tool_parsing;
        let prompt_tools = if expose_text_tools {
            sub_tools
        } else {
            empty_tools
        };
        let has_shell = prompt_tools.iter().any(|t| t.name() == "shell");
        let shell_policy = if has_shell {
            "## Shell Policy\n\n\
             - Prefer non-destructive commands. Use `trash` over `rm` where possible.\n\
             - Do not run commands that exfiltrate data or modify system-critical paths.\n\
             - Avoid interactive commands that block on stdin.\n\
             - Quote paths that may contain spaces."
                .to_string()
        } else {
            String::new()
        };

        // Build structured operational context using SystemPromptBuilder sections.
        let ctx = PromptContext {
            workspace_dir,
            agent_workspace_dir: workspace_dir,
            // This delegate operational prompt omits IdentitySection, so no
            // personality/persona overlay applies here.
            persona_bundles: &[],
            model_name,
            tools: prompt_tools,
            skills: &skills,
            skills_prompt_mode: zeroclaw_config::schema::SkillsPromptInjectionMode::Full,
            identity_config: None,
            dispatcher_instructions: "",
            sends_native_tool_specs: sends_native_tool_specs && !prompt_tools.is_empty(),

            security_summary: None,
            autonomy_level: crate::security::AutonomyLevel::default(),
        };

        let builder = SystemPromptBuilder::default()
            .add_section(Box::new(crate::agent::prompt::ToolsSection))
            .add_section(Box::new(crate::agent::prompt::SafetySection))
            .add_section(Box::new(crate::agent::prompt::SkillsSection))
            .add_section(Box::new(crate::agent::prompt::WorkspaceSection))
            .add_section(Box::new(crate::agent::prompt::DateTimeSection));

        let mut enriched = builder.build(&ctx).unwrap_or_default();

        if !shell_policy.is_empty() {
            enriched.push_str(&shell_policy);
            enriched.push_str("\n\n");
        }

        // Append the per-agent identity files from the target
        // sub-agent's own workspace dir. Each missing file is silently
        // skipped — the operator may not have authored every file.
        // Skipped entirely when no `root_config` is attached (legacy
        // unit-test constructors); production paths always attach it.
        if let Some(target_workspace) = self.agent_workspace(agent_alias) {
            let identity_files = [
                "AGENTS.md",
                "SOUL.md",
                "IDENTITY.md",
                "USER.md",
                "BOOTSTRAP.md",
            ];
            for filename in identity_files {
                let path = target_workspace.join(filename);
                if let Ok(contents) = std::fs::read_to_string(&path) {
                    let trimmed = contents.trim();
                    if !trimmed.is_empty() {
                        enriched.push_str(trimmed);
                        enriched.push_str("\n\n");
                    }
                }
            }
        }

        let trimmed = enriched.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }

    async fn execute_agentic(
        &self,
        agent_name: &str,
        agent_config: &AliasedAgentConfig,
        provider_type: &str,
        model: &str,
        model_provider: &dyn ModelProvider,
        full_prompt: &str,
        temperature: Option<f64>,
    ) -> anyhow::Result<ToolResult> {
        let allowed_tools = self.resolve_allowed_tools(&agent_config.risk_profile);

        if allowed_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' is agentic but risk_profile '{}' has no allowed_tools",
                    agent_config.risk_profile
                )),
            });
        }

        let allowed = allowed_tools
            .iter()
            .map(|name: &String| name.trim())
            .filter(|name| !name.is_empty())
            .collect::<std::collections::HashSet<_>>();

        let sub_tools: Vec<Box<dyn Tool>> = {
            let parent_tools = self.parent_tools.read();
            parent_tools
                .iter()
                .filter(|tool| allowed.contains(tool.name()))
                .filter(|tool| tool.name() != "delegate")
                .map(|tool| Box::new(ToolArcRef::new(tool.clone())) as Box<dyn Tool>)
                .collect()
        };

        if sub_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' has no executable tools after filtering allowlist ({})",
                    allowed_tools.join(", ")
                )),
            });
        }

        let max_iterations = self.resolve_max_iterations(&agent_config.runtime_profile);

        // Build enriched system prompt with tools, skills, workspace, datetime context.
        let enriched_system_prompt = self.build_enriched_system_prompt(
            agent_name,
            agent_config,
            model,
            &sub_tools,
            &self.workspace_dir,
            model_provider.supports_native_tools(),
        );

        let mut history = Vec::new();
        if let Some(system_prompt) = enriched_system_prompt.as_ref() {
            history.push(ChatMessage::system(system_prompt.clone()));
        }
        history.push(ChatMessage::user(full_prompt.to_string()));

        // Forward this delegate's observer into the agentic sub-loop so the
        // sub-agent's typed llm.request/llm.response/tool.* events flow into the
        // trace (the legacy record! macro fires regardless, but the typed
        // Observer path is what carries model_provider + tokens — and what the
        // zc-delegate-stats reporter prefers). When `self.observer` is unset
        // (unit-test constructors), fall back to NoopObserver, preserving the
        // pre-port behaviour. Symmetric with the non-agentic path that emits
        // an `LlmResponse` event via `self.observer`.
        let noop_observer = NoopObserver;
        let sub_observer: &dyn Observer = match &self.observer {
            Some(o) => &**o,
            None => &noop_observer,
        };

        let agentic_timeout_secs = self
            .resolve_agentic_timeout_secs(&agent_config.runtime_profile)
            .unwrap_or(self.delegate_config.agentic_timeout_secs);
        // Forward the per-turn receipt scope from the parent loop so subagent
        // tool calls land in the same collector as the top-level turn. When
        // receipts are disabled (or no scope is set, e.g. CLI / background
        // delegate spawn) this resolves to `None` and the sub-loop runs
        // unsigned, matching the parent.
        let receipt_scope = crate::agent::tool_receipts::TOOL_LOOP_RECEIPT_CONTEXT
            .try_with(Clone::clone)
            .ok()
            .flatten();
        let receipt_generator = receipt_scope.as_ref().map(|s| &s.generator);
        let collected_receipts = receipt_scope.as_ref().map(|s| s.collector.as_ref());
        let result = tokio::time::timeout(
            Duration::from_secs(agentic_timeout_secs),
            run_tool_call_loop(
                model_provider,
                &mut history,
                &sub_tools,
                sub_observer,
                provider_type,
                model,
                temperature,
                true,
                None,
                "delegate",
                None,
                &self.multimodal_config,
                max_iterations,
                Some(self.cancellation_token.child_token()),
                None,
                None,
                &[],
                &[],
                None,
                None,
                &zeroclaw_config::schema::PacingConfig::default(),
                agent_config.resolved.strict_tool_parsing,
                agent_config.resolved.parallel_tools,
                0,    // max_tool_result_chars: inherit from parent config in future
                0,    // context_token_budget: 0 = disabled for subagents
                None, // shared_budget: TODO thread from parent in future
                None, // channel: delegate subagents don't support approval
                receipt_generator,
                collected_receipts,
            )
            .instrument(::zeroclaw_log::attribution_span!(
                &crate::agent::AgentAttribution(agent_name)
            )),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                let rendered = if response.trim().is_empty() {
                    "[Empty response]".to_string()
                } else {
                    response
                };

                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "[Agent '{agent_name}' ({provider_type}/{model}, agentic)]\n{rendered}",
                    ),
                    error: None,
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}")),
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' timed out after {agentic_timeout_secs}s"
                )),
            }),
        }
    }
}

struct ToolArcRef {
    inner: Arc<dyn Tool>,
}

impl ToolArcRef {
    fn new(inner: Arc<dyn Tool>) -> Self {
        Self { inner }
    }
}

impl ::zeroclaw_api::attribution::Attributable for ToolArcRef {
    fn role(&self) -> ::zeroclaw_api::attribution::Role {
        self.inner.role()
    }
    fn alias(&self) -> &str {
        self.inner.alias()
    }
}

#[async_trait]
impl Tool for ToolArcRef {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.inner.parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.inner.execute(args).await
    }
}

struct NoopObserver;

impl Observer for NoopObserver {
    fn record_event(&self, _event: &ObserverEvent) {}

    fn record_metric(&self, _metric: &ObserverMetric) {}

    fn name(&self) -> &str {
        "noop"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{AutonomyLevel, SecurityPolicy};
    use std::path::Path;
    use tokio::time::{Instant, sleep};
    use zeroclaw_config::schema::{
        DEFAULT_DELEGATE_AGENTIC_TIMEOUT_SECS, DEFAULT_DELEGATE_TIMEOUT_SECS,
    };
    use zeroclaw_providers::{ChatRequest, ChatResponse, ToolCall};

    zeroclaw_api::mock_tool_attribution!(EchoTool, FakeMcpTool);

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy::default())
    }

    fn security_allowing() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            delegation_policy: zeroclaw_config::autonomy::DelegationPolicy {
                mode: zeroclaw_config::autonomy::DelegationMode::Allow,
            },
            ..SecurityPolicy::default()
        })
    }

    fn sample_agents() -> HashMap<String, AliasedAgentConfig> {
        let mut agents = HashMap::new();
        agents.insert(
            "researcher".to_string(),
            AliasedAgentConfig {
                model_provider: "ollama.researcher".into(),
                ..Default::default()
            },
        );
        agents.insert(
            "coder".to_string(),
            AliasedAgentConfig {
                model_provider: "openrouter.coder".into(),
                ..Default::default()
            },
        );
        agents
    }

    async fn wait_for_terminal_background_result(
        workspace: &Path,
        task_id: &str,
    ) -> BackgroundDelegateResult {
        let result_path = workspace
            .join("delegate_results")
            .join(format!("{task_id}.json"));
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut last_result = None;

        loop {
            if let Ok(content) = std::fs::read_to_string(&result_path) {
                let result: BackgroundDelegateResult = serde_json::from_str(&content).unwrap();
                if result.status != BackgroundTaskStatus::Running {
                    return result;
                }
                last_result = Some(result);
            }

            if Instant::now() >= deadline {
                panic!(
                    "Background task {task_id} did not finish before timeout; last result: {last_result:?}"
                );
            }

            sleep(Duration::from_millis(50)).await;
        }
    }

    #[derive(Default)]
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
        }

        fn description(&self) -> &str {
            "Echoes the `value` argument."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            })
        }

        async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
            let value = args
                .get("value")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            Ok(ToolResult {
                success: true,
                output: format!("echo:{value}"),
                error: None,
            })
        }
    }

    struct OneToolThenFinalModelProvider;

    #[async_trait]
    impl ModelProvider for OneToolThenFinalModelProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            let has_tool_message = request.messages.iter().any(|m| m.role == "tool");
            if has_tool_message {
                Ok(ChatResponse {
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                })
            } else {
                Ok(ChatResponse {
                    text: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "echo_tool".to_string(),
                        arguments: "{\"value\":\"ping\"}".to_string(),
                        extra_content: None,
                    }],
                    usage: None,
                    reasoning_content: None,
                })
            }
        }
    }
    impl ::zeroclaw_api::attribution::Attributable for OneToolThenFinalModelProvider {
        fn role(&self) -> ::zeroclaw_api::attribution::Role {
            ::zeroclaw_api::attribution::Role::Provider(
                ::zeroclaw_api::attribution::ProviderKind::Model(
                    ::zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }
        fn alias(&self) -> &str {
            "OneToolThenFinalModelProvider"
        }
    }

    struct TextFallbackToolModelProvider;

    #[async_trait]
    impl ModelProvider for TextFallbackToolModelProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            Ok(ChatResponse {
                text: Some(
                    r#"<tool_call>{"name":"echo_tool","arguments":{"value":"ignored"}}</tool_call>"#
                        .to_string(),
                ),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            })
        }
    }
    impl ::zeroclaw_api::attribution::Attributable for TextFallbackToolModelProvider {
        fn role(&self) -> ::zeroclaw_api::attribution::Role {
            ::zeroclaw_api::attribution::Role::Provider(
                ::zeroclaw_api::attribution::ProviderKind::Model(
                    ::zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }
        fn alias(&self) -> &str {
            "TextFallbackToolModelProvider"
        }
    }

    struct InfiniteToolCallModelProvider;

    #[async_trait]
    impl ModelProvider for InfiniteToolCallModelProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            Ok(ChatResponse {
                text: None,
                tool_calls: vec![ToolCall {
                    id: "loop".to_string(),
                    name: "echo_tool".to_string(),
                    arguments: "{\"value\":\"x\"}".to_string(),
                    extra_content: None,
                }],
                usage: None,
                reasoning_content: None,
            })
        }
    }
    impl ::zeroclaw_api::attribution::Attributable for InfiniteToolCallModelProvider {
        fn role(&self) -> ::zeroclaw_api::attribution::Role {
            ::zeroclaw_api::attribution::Role::Provider(
                ::zeroclaw_api::attribution::ProviderKind::Model(
                    ::zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }
        fn alias(&self) -> &str {
            "InfiniteToolCallModelProvider"
        }
    }

    struct FailingModelProvider;

    #[async_trait]
    impl ModelProvider for FailingModelProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            Err(anyhow::Error::msg("model_provider boom"))
        }
    }
    impl ::zeroclaw_api::attribution::Attributable for FailingModelProvider {
        fn role(&self) -> ::zeroclaw_api::attribution::Role {
            ::zeroclaw_api::attribution::Role::Provider(
                ::zeroclaw_api::attribution::ProviderKind::Model(
                    ::zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }
        fn alias(&self) -> &str {
            "FailingModelProvider"
        }
    }

    fn agentic_agent_config() -> AliasedAgentConfig {
        AliasedAgentConfig {
            model_provider: "openrouter.agentic".into(),
            risk_profile: "agentic_test".to_string(),
            runtime_profile: "agentic_test".to_string(),
            ..Default::default()
        }
    }

    fn agentic_providers_models() -> HashMap<String, HashMap<String, ModelProviderConfig>> {
        let mut models: HashMap<String, HashMap<String, ModelProviderConfig>> = HashMap::new();
        models.entry("openrouter".to_string()).or_default().insert(
            "agentic".to_string(),
            ModelProviderConfig {
                model: Some("model-test".to_string()),
                temperature: Some(0.2),
                api_key: Some("delegate-test-credential".to_string()),
                ..Default::default()
            },
        );
        models
    }

    fn agentic_runtime_profiles(max_iterations: usize) -> HashMap<String, RuntimeProfileConfig> {
        let mut profiles = HashMap::new();
        profiles.insert(
            "agentic_test".to_string(),
            RuntimeProfileConfig {
                agentic: true,
                max_tool_iterations: max_iterations,
                ..Default::default()
            },
        );
        profiles
    }

    fn agentic_risk_profiles(allowed_tools: Vec<String>) -> HashMap<String, RiskProfileConfig> {
        let mut profiles = HashMap::new();
        profiles.insert(
            "agentic_test".to_string(),
            RiskProfileConfig {
                allowed_tools,
                ..Default::default()
            },
        );
        profiles
    }

    #[test]
    fn name_and_schema() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        assert_eq!(tool.name(), "delegate");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["prompt"].is_object());
        assert!(schema["properties"]["context"].is_object());
        assert!(schema["properties"]["background"].is_object());
        assert!(schema["properties"]["parallel"].is_object());
        assert!(schema["properties"]["action"].is_object());
        assert!(schema["properties"]["task_id"].is_object());
        // required is empty because different actions need different params
        let required = schema["required"].as_array().unwrap();
        assert!(required.is_empty());
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["properties"]["agent"]["minLength"], json!(1));
        assert_eq!(schema["properties"]["prompt"]["minLength"], json!(1));
    }

    #[test]
    fn description_not_empty() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn schema_lists_agent_names() {
        let tool = DelegateTool::new(sample_agents(), None, security_allowing());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("researcher") || desc.contains("coder"));
    }

    #[test]
    fn schema_roster_filtered_by_delegation_policy() {
        // When delegation is permitted, every configured agent (minus the
        // caller) is advertised — reachability is gated by shared risk
        // profile at delegation time, not by a per-agent roster allow-list.
        let tool = DelegateTool::new(sample_agents(), None, security_allowing());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("researcher"));
        assert!(desc.contains("coder"));

        // When delegation is forbidden, the roster is empty.
        let forbidden =
            DelegateTool::new(sample_agents(), None, Arc::new(SecurityPolicy::default()));
        let forbidden_schema = forbidden.parameters_schema();
        let forbidden_desc = forbidden_schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(!forbidden_desc.contains("researcher"));
        assert!(!forbidden_desc.contains("coder"));
    }

    #[test]
    fn schema_roster_lists_only_same_risk_profile_peers() {
        // Three agents: two on "alpha", one on "beta". Caller is on "alpha".
        let mut agents = HashMap::new();
        agents.insert(
            "alpha_peer".to_string(),
            AliasedAgentConfig {
                risk_profile: "alpha".into(),
                ..Default::default()
            },
        );
        agents.insert(
            "alpha_self".to_string(),
            AliasedAgentConfig {
                risk_profile: "alpha".into(),
                ..Default::default()
            },
        );
        agents.insert(
            "beta_outsider".to_string(),
            AliasedAgentConfig {
                risk_profile: "beta".into(),
                ..Default::default()
            },
        );

        // Caller on "alpha" with delegation allowed; it owns "alpha_self".
        let mut policy = SecurityPolicy {
            delegation_policy: zeroclaw_config::autonomy::DelegationPolicy {
                mode: zeroclaw_config::autonomy::DelegationMode::Allow,
            },
            ..SecurityPolicy::default()
        };
        policy.risk_profile_name = "alpha".into();
        let mut tool = DelegateTool::new(agents, None, Arc::new(policy));
        tool.caller_alias = "alpha_self".to_string();

        let desc = tool.parameters_schema()["properties"]["agent"]["description"]
            .as_str()
            .unwrap()
            .to_string();

        // Same-profile peer is listed.
        assert!(desc.contains("alpha_peer"), "{desc}");
        // Delegator excludes itself.
        assert!(!desc.contains("alpha_self"), "{desc}");
        // Off-profile agent is excluded.
        assert!(!desc.contains("beta_outsider"), "{desc}");
    }

    #[test]
    fn schema_excludes_caller_alias_from_roster() {
        // An agent must never be offered itself as a delegation target,
        // even when the delegation_policy would otherwise permit it.
        let tool = DelegateTool::new(sample_agents(), None, security_allowing())
            .with_caller_alias("researcher");
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(!desc.contains("researcher"));
        assert!(desc.contains("coder"));
    }

    #[test]
    fn schema_empty_roster_when_delegation_forbidden() {
        // Default policy forbids delegation, so no configured agent
        // should be advertised.
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("none configured"));
    }

    #[tokio::test]
    async fn missing_agent_param() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool.execute(json!({"prompt": "test"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn missing_prompt_param() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool.execute(json!({"agent": "researcher"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unknown_agent_returns_error() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "nonexistent", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown agent"));
    }

    #[tokio::test]
    async fn depth_limit_enforced() {
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 3);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("depth limit"));
    }

    #[tokio::test]
    async fn depth_limit_at_default_max() {
        // Default max_depth is 3; at depth=3 the agent should be blocked.
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 3);
        let result = tool
            .execute(json!({"agent": "coder", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("depth limit"));
    }

    #[test]
    fn empty_agents_schema() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("none configured"));
    }

    #[tokio::test]
    async fn invalid_provider_returns_error() {
        let mut agents = HashMap::new();
        agents.insert(
            "broken".to_string(),
            AliasedAgentConfig {
                model_provider: "totally-invalid-provider.default".into(),
                ..Default::default()
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({"agent": "broken", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result
                .error
                .unwrap()
                .contains("Failed to create model_provider")
        );
    }

    #[tokio::test]
    async fn blank_agent_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "  ", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn blank_prompt_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "  \t  "}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn whitespace_agent_name_trimmed_and_found() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        // " researcher " with surrounding whitespace — after trim becomes "researcher"
        let result = tool
            .execute(json!({"agent": " researcher ", "prompt": "test"}))
            .await
            .unwrap();
        // Should find "researcher" after trim — will fail at model_provider level
        // since ollama isn't running, but must NOT get "Unknown agent".
        assert!(
            result.error.is_none()
                || !result
                    .error
                    .as_deref()
                    .unwrap_or("")
                    .contains("Unknown agent")
        );
    }

    #[tokio::test]
    async fn delegation_blocked_in_readonly_mode() {
        let readonly = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = DelegateTool::new(sample_agents(), None, readonly);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("read-only mode")
        );
    }

    #[tokio::test]
    async fn delegation_blocked_when_rate_limited() {
        let limited = Arc::new(SecurityPolicy {
            max_actions_per_hour: 0,
            ..SecurityPolicy::default()
        });
        let tool = DelegateTool::new(sample_agents(), None, limited);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("Rate limit exceeded")
        );
    }

    #[tokio::test]
    async fn delegate_context_is_prepended_to_prompt() {
        let mut agents = HashMap::new();
        agents.insert(
            "tester".to_string(),
            AliasedAgentConfig {
                model_provider: "invalid-for-test.default".into(),
                ..Default::default()
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({
                "agent": "tester",
                "prompt": "do something",
                "context": "some context data"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("Failed to create model_provider")
        );
    }

    #[tokio::test]
    async fn delegate_empty_context_omits_prefix() {
        let mut agents = HashMap::new();
        agents.insert(
            "tester".to_string(),
            AliasedAgentConfig {
                model_provider: "invalid-for-test.default".into(),
                ..Default::default()
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({
                "agent": "tester",
                "prompt": "do something",
                "context": ""
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("Failed to create model_provider")
        );
    }

    #[test]
    fn delegate_depth_construction() {
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 5);
        assert_eq!(tool.depth, 5);
    }

    #[tokio::test]
    async fn delegate_no_agents_configured() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security());
        let result = tool
            .execute(json!({"agent": "any", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("none configured"));
    }

    #[tokio::test]
    async fn agentic_mode_rejects_empty_allowed_tools() {
        let mut agents = HashMap::new();
        agents.insert("agentic".to_string(), agentic_agent_config());

        let tool = DelegateTool::new(agents, None, test_security())
            .with_providers_models(agentic_providers_models())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(Vec::new()));
        let result = tool
            .execute(json!({"agent": "agentic", "prompt": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("has no allowed_tools"),
            "got: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn agentic_mode_rejects_unmatched_allowed_tools() {
        let mut agents = HashMap::new();
        agents.insert("agentic".to_string(), agentic_agent_config());

        let allowed = vec!["missing_tool".to_string()];
        let tool = DelegateTool::new(agents, None, test_security())
            .with_providers_models(agentic_providers_models())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(allowed))
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));
        let result = tool
            .execute(json!({"agent": "agentic", "prompt": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("no executable tools")
        );
    }

    #[tokio::test]
    async fn execute_agentic_runs_tool_call_loop_with_filtered_tools() {
        let config = agentic_agent_config();
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(vec!["echo_tool".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(vec![
                Arc::new(EchoTool),
                Arc::new(DelegateTool::new(HashMap::new(), None, test_security())),
            ])));

        let model_provider = OneToolThenFinalModelProvider;
        let result = tool
            .execute_agentic(
                "agentic",
                &config,
                "openrouter",
                "model-test",
                &model_provider,
                "run",
                Some(0.2),
            )
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("(openrouter/model-test, agentic)"));
        assert!(result.output.contains("done"));
    }

    #[tokio::test]
    async fn execute_agentic_strict_tool_parsing_uses_target_agent_policy() {
        let mut config = agentic_agent_config();
        config.resolved.strict_tool_parsing = true;
        let prompt_tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(vec!["echo_tool".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));

        let prompt = tool
            .build_enriched_system_prompt(
                "agentic",
                &config,
                "model-test",
                &prompt_tools,
                Path::new("/tmp"),
                false,
            )
            .expect("prompt should render");
        assert!(
            !prompt.contains("## Tools"),
            "strict delegate prompt should not advertise text tool instructions"
        );
        assert!(
            !prompt.contains("echo_tool"),
            "strict delegate prompt should hide text-only tool schemas"
        );

        let model_provider = TextFallbackToolModelProvider;
        let result = tool
            .execute_agentic(
                "agentic",
                &config,
                "openrouter",
                "model-test",
                &model_provider,
                "run",
                Some(0.2),
            )
            .await
            .unwrap();

        assert!(result.success);
        assert!(
            result.output.contains("<tool_call>"),
            "strict subagent should return fallback-looking text unchanged"
        );
        assert!(
            !result.output.contains("echo:ignored"),
            "strict subagent must not execute text fallback tool calls"
        );
    }

    #[tokio::test]
    async fn execute_agentic_excludes_delegate_even_if_allowlisted() {
        let config = agentic_agent_config();
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(vec!["delegate".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(DelegateTool::new(
                HashMap::new(),
                None,
                test_security(),
            ))])));

        let model_provider = OneToolThenFinalModelProvider;
        let result = tool
            .execute_agentic(
                "agentic",
                &config,
                "openrouter",
                "model-test",
                &model_provider,
                "run",
                Some(0.2),
            )
            .await
            .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("no executable tools")
        );
    }

    #[tokio::test]
    async fn execute_agentic_respects_max_iterations() {
        let config = agentic_agent_config();
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(2))
            .with_risk_profiles(agentic_risk_profiles(vec!["echo_tool".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));

        let model_provider = InfiniteToolCallModelProvider;
        let result = tool
            .execute_agentic(
                "agentic",
                &config,
                "openrouter",
                "model-test",
                &model_provider,
                "run",
                Some(0.2),
            )
            .await
            .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("maximum tool iterations (2)")
        );
    }

    #[tokio::test]
    async fn execute_agentic_forwards_receipt_scope_into_subagent_loop() {
        // Receipt forwarding through the delegate sub-loop is the activation
        // pass for #6182's delegate.rs:1184 acceptance criterion. With
        // `TOOL_LOOP_RECEIPT_CONTEXT` scoped, every sub-tool call inside the
        // delegate must produce a receipt that lands in the same per-turn
        // collector the parent passed in. Without the task-local read in
        // `execute_sync` this test fails: the collector stays empty because
        // the sub-loop runs unsigned with `None, None` for the receipt args.
        use crate::agent::tool_receipts::{
            ReceiptGenerator, ReceiptScope, TOOL_LOOP_RECEIPT_CONTEXT,
        };

        let config = agentic_agent_config();
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(vec!["echo_tool".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));

        let collector: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let scope = ReceiptScope {
            generator: ReceiptGenerator::new(),
            collector: Arc::clone(&collector),
        };

        let model_provider = OneToolThenFinalModelProvider;
        let result = TOOL_LOOP_RECEIPT_CONTEXT
            .scope(Some(scope), async {
                tool.execute_agentic(
                    "agentic",
                    &config,
                    "test-provider",
                    "test-model",
                    &model_provider,
                    "run",
                    Some(0.2),
                )
                .await
            })
            .await
            .unwrap();

        assert!(
            result.success,
            "delegate sub-loop must complete: {result:?}"
        );
        let receipts = collector.lock().unwrap();
        assert_eq!(
            receipts.len(),
            1,
            "expected exactly one receipt for the single echo_tool sub-call, got: {:?}",
            receipts.as_slice()
        );
        assert!(
            receipts[0].starts_with("echo_tool: zc-receipt-"),
            "sub-tool receipt must be tagged with the tool name and a zc-receipt- HMAC token, got: {}",
            receipts[0]
        );
    }

    #[tokio::test]
    async fn delegate_spawn_helper_forwards_session_key() {
        let seen = TOOL_LOOP_SESSION_KEY
            .scope(Some("channel_session".to_string()), async {
                let session_key = current_tool_loop_session_key();
                zeroclaw_spawn::spawn!(async move {
                    scope_delegate_session_key(session_key, async {
                        current_tool_loop_session_key()
                    })
                    .await
                })
                .await
                .unwrap()
            })
            .await;

        assert_eq!(seen.as_deref(), Some("channel_session"));
    }

    #[tokio::test]
    async fn execute_agentic_emits_no_receipts_when_scope_absent() {
        // Backward-compat for callers without a scoped receipt context (CLI,
        // background spawn that does not forward scope, tests). The sub-loop
        // must run unsigned and the agent output must not carry a
        // `[receipt: ` trailer.
        let config = agentic_agent_config();
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(vec!["echo_tool".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));

        let model_provider = OneToolThenFinalModelProvider;
        let result = tool
            .execute_agentic(
                "agentic",
                &config,
                "test-provider",
                "test-model",
                &model_provider,
                "run",
                Some(0.2),
            )
            .await
            .unwrap();

        assert!(result.success);
        assert!(
            !result.output.contains("[receipt: "),
            "no receipt trailer must appear in agent output when receipts are disabled, got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn execute_agentic_propagates_provider_errors() {
        let config = agentic_agent_config();
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(vec!["echo_tool".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));

        let model_provider = FailingModelProvider;
        let result = tool
            .execute_agentic(
                "agentic",
                &config,
                "openrouter",
                "model-test",
                &model_provider,
                "run",
                Some(0.2),
            )
            .await
            .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("model_provider boom")
        );
    }

    /// MCP tools pushed into the shared parent_tools handle after DelegateTool
    /// construction must be visible to the sub-agent tool list.
    #[derive(Default)]
    struct FakeMcpTool;

    #[async_trait]
    impl Tool for FakeMcpTool {
        fn name(&self) -> &str {
            "mcp_fake"
        }

        fn description(&self) -> &str {
            "Fake MCP tool for testing."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                success: true,
                output: "mcp_fake_output".into(),
                error: None,
            })
        }
    }

    struct McpToolThenFinalModelProvider;

    #[async_trait]
    impl ModelProvider for McpToolThenFinalModelProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            let has_tool_message = request.messages.iter().any(|m| m.role == "tool");
            if has_tool_message {
                Ok(ChatResponse {
                    text: Some("mcp done".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                })
            } else {
                Ok(ChatResponse {
                    text: None,
                    tool_calls: vec![ToolCall {
                        id: "call_mcp".to_string(),
                        name: "mcp_fake".to_string(),
                        arguments: "{}".to_string(),
                        extra_content: None,
                    }],
                    usage: None,
                    reasoning_content: None,
                })
            }
        }
    }
    impl ::zeroclaw_api::attribution::Attributable for McpToolThenFinalModelProvider {
        fn role(&self) -> ::zeroclaw_api::attribution::Role {
            ::zeroclaw_api::attribution::Role::Provider(
                ::zeroclaw_api::attribution::ProviderKind::Model(
                    ::zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }
        fn alias(&self) -> &str {
            "McpToolThenFinalModelProvider"
        }
    }

    #[tokio::test]
    async fn mcp_tools_included_in_subagent_tool_list() {
        // Build DelegateTool with NO parent tools initially
        let config = agentic_agent_config();
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(agentic_runtime_profiles(10))
            .with_risk_profiles(agentic_risk_profiles(vec!["mcp_fake".to_string()]))
            .with_parent_tools(Arc::new(RwLock::new(Vec::new())));

        // Simulate late MCP tool injection via the shared handle
        let handle = tool.parent_tools_handle();
        handle.write().push(Arc::new(FakeMcpTool));

        let model_provider = McpToolThenFinalModelProvider;
        let result = tool
            .execute_agentic(
                "agentic",
                &config,
                "openrouter",
                "model-test",
                &model_provider,
                "run mcp",
                Some(0.2),
            )
            .await
            .unwrap();

        assert!(result.success, "Expected success, got: {:?}", result.error);
        assert!(
            result.output.contains("mcp done"),
            "Expected output containing 'mcp done', got: {}",
            result.output
        );
    }

    #[test]
    fn enriched_prompt_includes_tools_workspace_date() {
        let config = AliasedAgentConfig {
            model_provider: "openrouter.test".into(),
            ..Default::default()
        };

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_enrich_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_workspace_dir(workspace.clone());

        let prompt = tool
            .build_enriched_system_prompt("alpha", &config, "test-model", &tools, &workspace, false)
            .unwrap();

        assert!(prompt.contains("## Tools"), "should contain tools section");
        assert!(prompt.contains("echo_tool"), "should list allowed tools");
        assert!(
            prompt.contains("## Workspace"),
            "should contain workspace section"
        );
        assert!(
            prompt.contains(&workspace.display().to_string()),
            "should contain workspace path"
        );
        assert!(
            prompt.contains("## CRITICAL CONTEXT: CURRENT DATE"),
            "should contain date section"
        );
        assert!(!prompt.contains("CURRENT DATE & TIME"));
        assert!(!prompt.contains("Time:"));
        assert!(!prompt.contains("ISO 8601:"));
        // Identity files come from the target sub-agent's per-agent
        // workspace dir. The test's install_root is unset, so no
        // identity files exist for the dummy alias — the prompt still
        // contains the structural sections verified above, which is
        // the load-bearing assertion.

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[test]
    fn enriched_prompt_includes_shell_policy_when_shell_present() {
        let config = AliasedAgentConfig::default();

        struct MockShellTool;
        impl ::zeroclaw_api::attribution::Attributable for MockShellTool {
            fn role(&self) -> ::zeroclaw_api::attribution::Role {
                ::zeroclaw_api::attribution::Role::Tool(
                    ::zeroclaw_api::attribution::ToolKind::Shell,
                )
            }
            fn alias(&self) -> &str {
                <Self as Tool>::name(self)
            }
        }
        #[async_trait]
        impl Tool for MockShellTool {
            fn name(&self) -> &str {
                "shell"
            }
            fn description(&self) -> &str {
                "Execute shell commands"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                json!({"type": "object"})
            }
            async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    success: true,
                    output: String::new(),
                    error: None,
                })
            }
        }

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockShellTool)];
        let workspace = std::env::temp_dir();

        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_workspace_dir(workspace.to_path_buf());

        let prompt = tool
            .build_enriched_system_prompt("alpha", &config, "test-model", &tools, &workspace, false)
            .unwrap();

        assert!(
            prompt.contains("## Shell Policy"),
            "should contain shell policy when shell tool is present"
        );
    }

    #[test]
    fn parent_tools_handle_returns_shared_reference() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security()).with_parent_tools(
            Arc::new(RwLock::new(vec![Arc::new(EchoTool) as Arc<dyn Tool>])),
        );

        let handle = tool.parent_tools_handle();
        assert_eq!(handle.read().len(), 1);

        // Push a new tool via the handle
        handle.write().push(Arc::new(FakeMcpTool));
        assert_eq!(handle.read().len(), 2);
    }

    // ── Configurable timeout tests ──────────────────────────────────

    #[test]
    fn delegate_timeout_defaults_come_from_delegate_config() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_delegate_config(DelegateToolConfig::default());
        assert_eq!(
            tool.delegate_config.timeout_secs,
            DEFAULT_DELEGATE_TIMEOUT_SECS
        );
        assert_eq!(
            tool.delegate_config.agentic_timeout_secs,
            DEFAULT_DELEGATE_AGENTIC_TIMEOUT_SECS
        );
    }

    #[test]
    fn enriched_prompt_omits_shell_policy_without_shell_tool() {
        let config = AliasedAgentConfig::default();

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let workspace = std::env::temp_dir();

        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_workspace_dir(workspace.to_path_buf());

        let prompt = tool
            .build_enriched_system_prompt("alpha", &config, "test-model", &tools, &workspace, false)
            .unwrap();

        assert!(
            !prompt.contains("## Shell Policy"),
            "should not contain shell policy when shell tool is absent"
        );
    }

    #[test]
    fn config_validation_accepts_minimal_agent() {
        let mut config = zeroclaw_config::schema::Config::default();
        // model_provider must reference a real entry under
        // providers.models — the validator (correctly) rejects dangling refs.
        config.providers.models.ollama.insert(
            "default".into(),
            zeroclaw_config::schema::OllamaModelProviderConfig::default(),
        );
        config.risk_profiles.insert(
            "default".into(),
            zeroclaw_config::schema::RiskProfileConfig::default(),
        );
        config.agents.insert(
            "ok".into(),
            AliasedAgentConfig {
                model_provider: "ollama.default".into(),
                risk_profile: "default".into(),
                ..Default::default()
            },
        );
        assert!(
            config.validate().is_ok(),
            "validate: {:?}",
            config.validate()
        );
    }

    #[test]
    fn enriched_prompt_loads_skills_from_scoped_directory() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_skills_test_{}",
            uuid::Uuid::new_v4()
        ));
        let scoped_skills_dir = workspace.join("skills/code-review");
        std::fs::create_dir_all(scoped_skills_dir.join("lint-check")).unwrap();
        std::fs::write(
            scoped_skills_dir.join("lint-check/SKILL.toml"),
            "[skill]\nname = \"lint-check\"\ndescription = \"Run lint checks\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();

        let config = AliasedAgentConfig {
            skill_bundles: vec!["code_review".to_string()],
            ..Default::default()
        };

        let mut skill_bundles = HashMap::new();
        skill_bundles.insert(
            "code_review".to_string(),
            SkillBundleConfig {
                directory: Some("skills/code-review".to_string()),
                ..Default::default()
            },
        );

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];

        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_skill_bundles(skill_bundles)
            .with_workspace_dir(workspace.clone());

        let prompt = tool
            .build_enriched_system_prompt("alpha", &config, "test-model", &tools, &workspace, false)
            .unwrap();

        assert!(
            prompt.contains("lint-check"),
            "should contain skills from scoped directory"
        );

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[test]
    fn enriched_prompt_falls_back_to_default_skills_dir() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_fallback_test_{}",
            uuid::Uuid::new_v4()
        ));
        let default_skills_dir = workspace.join("skills");
        std::fs::create_dir_all(default_skills_dir.join("deploy")).unwrap();
        std::fs::write(
            default_skills_dir.join("deploy/SKILL.toml"),
            "[skill]\nname = \"deploy\"\ndescription = \"Deploy safely\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();

        let config = AliasedAgentConfig::default();

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];

        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_workspace_dir(workspace.clone());

        let prompt = tool
            .build_enriched_system_prompt("alpha", &config, "test-model", &tools, &workspace, false)
            .unwrap();

        assert!(
            prompt.contains("deploy"),
            "should contain skills from default workspace skills/ directory"
        );

        let _ = std::fs::remove_dir_all(workspace);
    }

    // ── Background and Parallel execution tests ─────────────────────

    #[tokio::test]
    async fn background_delegation_returns_task_id() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_bg_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        let result = tool
            .execute(json!({
                "agent": "researcher",
                "prompt": "test background",
                "background": true
            }))
            .await
            .unwrap();

        // The agent will fail at model_provider level (ollama not running),
        // but the background task should be spawned and return a task_id.
        assert!(result.success);
        assert!(result.output.contains("task_id:"));
        assert!(result.output.contains("Background task started"));

        // Wait a moment for the background task to write its result
        tokio::time::sleep(Duration::from_millis(200)).await;

        // The results directory should exist
        assert!(workspace.join("delegate_results").exists());

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn background_unknown_agent_rejected() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_bg_unknown_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        let result = tool
            .execute(json!({
                "agent": "nonexistent",
                "prompt": "test",
                "background": true
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown agent"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn check_result_missing_task_id() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_check_noid_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        let result = tool.execute(json!({"action": "check_result"})).await;

        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn check_result_nonexistent_task() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_check_miss_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        // Use a valid UUID format that doesn't correspond to any real task
        let fake_uuid = uuid::Uuid::new_v4().to_string();
        let result = tool
            .execute(json!({
                "action": "check_result",
                "task_id": fake_uuid
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("No result found"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn list_results_empty() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_list_empty_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        let result = tool
            .execute(json!({"action": "list_results"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("No background delegate results"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn parallel_empty_list_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({
                "parallel": [],
                "prompt": "test"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("at least one agent"));
    }

    #[tokio::test]
    async fn parallel_unknown_agent_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({
                "parallel": ["researcher", "nonexistent"],
                "prompt": "test"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown agent"));
    }

    #[tokio::test]
    async fn parallel_missing_prompt_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({
                "parallel": ["researcher"]
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unknown_action_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"action": "invalid_action"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown action"));
    }

    #[tokio::test]
    async fn cancel_task_nonexistent() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_cancel_miss_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        // Use a valid UUID format that doesn't correspond to any real task
        let fake_uuid = uuid::Uuid::new_v4().to_string();
        let result = tool
            .execute(json!({
                "action": "cancel_task",
                "task_id": fake_uuid
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("No task found"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[test]
    fn cancellation_token_accessor() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let token = tool.cancellation_token();
        assert!(!token.is_cancelled());

        tool.cancel_all_background_tasks();
        assert!(token.is_cancelled());
    }

    #[test]
    fn with_cancellation_token_replaces_default() {
        let custom_token = CancellationToken::new();
        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_cancellation_token(custom_token.clone());

        assert!(!tool.cancellation_token().is_cancelled());
        custom_token.cancel();
        assert!(tool.cancellation_token().is_cancelled());
    }

    #[tokio::test]
    async fn background_task_result_persisted_to_disk() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_bg_persist_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());

        let result = tool
            .execute(json!({
                "agent": "researcher",
                "prompt": "persistence test",
                "background": true
            }))
            .await
            .unwrap();

        assert!(result.success);

        // Extract task_id from output
        let task_id = result
            .output
            .lines()
            .find(|l| l.starts_with("task_id:"))
            .unwrap()
            .trim_start_matches("task_id: ")
            .trim();

        // Check that the result file exists
        let result_path = workspace
            .join("delegate_results")
            .join(format!("{task_id}.json"));
        assert!(
            result_path.exists(),
            "Result file should exist at {result_path:?}"
        );

        // Read and parse the result
        let bg_result = wait_for_terminal_background_result(&workspace, task_id).await;
        assert_eq!(bg_result.task_id, task_id);
        assert_eq!(bg_result.agent, "researcher");
        // The task will have failed because ollama isn't running, but it should be persisted
        assert!(
            bg_result.status == BackgroundTaskStatus::Completed
                || bg_result.status == BackgroundTaskStatus::Failed
        );
        assert!(bg_result.finished_at.is_some());

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn check_result_retrieves_persisted_background_result() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_check_retrieve_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());

        // Start background task
        let result = tool
            .execute(json!({
                "agent": "researcher",
                "prompt": "retrieval test",
                "background": true
            }))
            .await
            .unwrap();

        let task_id = result
            .output
            .lines()
            .find(|l| l.starts_with("task_id:"))
            .unwrap()
            .trim_start_matches("task_id: ")
            .trim()
            .to_string();

        // Wait for background task
        let _ = wait_for_terminal_background_result(&workspace, &task_id).await;

        // Check result
        let check = tool
            .execute(json!({
                "action": "check_result",
                "task_id": task_id
            }))
            .await
            .unwrap();

        // The output should contain the serialized result
        assert!(check.output.contains(&task_id));
        assert!(check.output.contains("researcher"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn list_results_includes_background_tasks() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_list_tasks_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());

        // Start a background task
        let result = tool
            .execute(json!({
                "agent": "researcher",
                "prompt": "list test",
                "background": true
            }))
            .await
            .unwrap();
        assert!(result.success);
        let task_id = result
            .output
            .lines()
            .find(|l| l.starts_with("task_id:"))
            .unwrap()
            .trim_start_matches("task_id: ")
            .trim();

        // Wait for task to complete
        let _ = wait_for_terminal_background_result(&workspace, task_id).await;

        // List results
        let list = tool
            .execute(json!({"action": "list_results"}))
            .await
            .unwrap();

        assert!(list.success);
        assert!(list.output.contains("researcher"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn default_action_is_delegate() {
        // Calling without action should behave like "delegate"
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        // Should proceed to delegation (will fail at model_provider since ollama isn't running)
        // but should NOT fail with "Unknown action" error
        assert!(
            result.error.is_none()
                || !result
                    .error
                    .as_deref()
                    .unwrap_or("")
                    .contains("Unknown action")
        );
    }

    #[tokio::test]
    async fn check_result_rejects_path_traversal() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_traversal_check_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        let result = tool
            .execute(json!({
                "action": "check_result",
                "task_id": "../../etc/passwd"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("Invalid task_id"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn cancel_task_rejects_path_traversal() {
        let workspace = std::env::temp_dir().join(format!(
            "zeroclaw_delegate_traversal_cancel_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();

        let tool = DelegateTool::new(sample_agents(), None, test_security())
            .with_workspace_dir(workspace.clone());
        let result = tool
            .execute(json!({
                "action": "cancel_task",
                "task_id": "../../../etc/shadow"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("Invalid task_id"));

        let _ = std::fs::remove_dir_all(workspace);
    }

    fn config_with_two_agents(
        caller_alias: &str,
        caller_max_actions: u32,
        target_alias: &str,
        target_max_actions: u32,
    ) -> Arc<zeroclaw_config::schema::Config> {
        use zeroclaw_config::autonomy::{DelegationMode, DelegationPolicy};
        use zeroclaw_config::schema::{
            AliasedAgentConfig, Config, RiskProfileConfig, RuntimeProfileConfig,
        };
        let mut config = Config::default();
        // The caller delegates from the `narrow` profile, so that profile must
        // authorize the target alias; without it the delegation_policy gate
        // rejects before the escalation/narrowing checks under test are reached.
        config.risk_profiles.insert(
            "narrow".to_string(),
            RiskProfileConfig {
                delegation_policy: DelegationPolicy {
                    mode: DelegationMode::Allow,
                },
                ..RiskProfileConfig::default()
            },
        );
        config
            .risk_profiles
            .insert("wide".to_string(), RiskProfileConfig::default());
        config.runtime_profiles.insert(
            "narrow".to_string(),
            RuntimeProfileConfig {
                max_actions_per_hour: caller_max_actions,
                ..RuntimeProfileConfig::default()
            },
        );
        config.runtime_profiles.insert(
            "wide".to_string(),
            RuntimeProfileConfig {
                max_actions_per_hour: target_max_actions,
                ..RuntimeProfileConfig::default()
            },
        );
        let pick = |above: bool| if above { "wide" } else { "narrow" }.to_string();
        config.agents.insert(
            caller_alias.to_string(),
            AliasedAgentConfig {
                risk_profile: "narrow".to_string(),
                runtime_profile: "narrow".to_string(),
                model_provider: "ollama.caller".into(),
                ..AliasedAgentConfig::default()
            },
        );
        config.agents.insert(
            target_alias.to_string(),
            AliasedAgentConfig {
                risk_profile: pick(target_max_actions > caller_max_actions),
                runtime_profile: pick(target_max_actions > caller_max_actions),
                model_provider: "ollama.target".into(),
                ..AliasedAgentConfig::default()
            },
        );
        Arc::new(config)
    }

    #[tokio::test]
    async fn delegate_refuses_broader_target_cross_profile() {
        // caller(narrow, max_actions=5) is authorized to delegate, but target
        // resolves onto the wider profile (max_actions=50). Delegation is
        // narrowing-only, so a BROADER target is a privilege escalation and
        // must be refused (the no-escalation invariant).
        let config = config_with_two_agents("caller", 5, "target", 50);
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, caller_policy)
            .with_root_config(config.clone())
            .with_caller_alias("caller");

        let err = tool
            .policy_for_target("target")
            .expect_err("a broader cross-profile target must be rejected");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("narrowing-only") && chain.contains("escalate"),
            "expected an escalation/narrowing rejection, got: {chain}"
        );
    }

    #[tokio::test]
    async fn delegate_target_inherits_caller_action_tracker() {
        let config = config_with_two_agents("caller", 5, "target", 5);
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, Arc::clone(&caller_policy))
            .with_root_config(config.clone());

        let bucket_key = "shared-budget-test";
        let max = 2u32;
        for _ in 0..max {
            assert!(
                caller_policy.tracker.record_within(bucket_key, max),
                "caller's first {max} actions fit within the shared budget"
            );
        }

        let target_policy = tool
            .policy_for_target("target")
            .expect("non-escalating target resolves");
        assert!(
            !target_policy.tracker.record_within(bucket_key, max),
            "delegated target must consume from the caller's bucket; spawning the target should not reset the budget"
        );
    }

    #[tokio::test]
    async fn delegate_without_root_config_falls_back_to_caller_policy() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let resolved = tool
            .policy_for_target("researcher")
            .expect("fallback path returns caller policy unchanged");
        assert!(
            Arc::ptr_eq(&resolved, &tool.security),
            "without root_config the helper returns the caller's Arc verbatim"
        );
    }

    /// Build a config where `caller` (`broad` profile) is authorized to
    /// delegate to `target`, but `target` sits on a different (`narrow`)
    /// profile. Delegation requires caller and target to share a risk
    /// profile, so this exercises the same-profile rejection gate.
    fn config_with_narrowed_target() -> Arc<zeroclaw_config::schema::Config> {
        use zeroclaw_config::autonomy::{DelegationMode, DelegationPolicy};
        use zeroclaw_config::schema::{AliasedAgentConfig, Config, RiskProfileConfig};
        let mut config = Config::default();
        config.risk_profiles.insert(
            "broad".to_string(),
            RiskProfileConfig {
                allowed_commands: vec!["git".into(), "cargo".into()],
                delegation_policy: DelegationPolicy {
                    mode: DelegationMode::Allow,
                },
                ..RiskProfileConfig::default()
            },
        );
        config.risk_profiles.insert(
            "narrow".to_string(),
            RiskProfileConfig {
                allowed_commands: vec!["git".into()],
                ..RiskProfileConfig::default()
            },
        );
        config.agents.insert(
            "caller".to_string(),
            AliasedAgentConfig {
                risk_profile: "broad".to_string(),
                model_provider: "ollama.caller".into(),
                ..AliasedAgentConfig::default()
            },
        );
        config.agents.insert(
            "target".to_string(),
            AliasedAgentConfig {
                risk_profile: "narrow".to_string(),
                model_provider: "ollama.target".into(),
                ..AliasedAgentConfig::default()
            },
        );
        Arc::new(config)
    }

    #[tokio::test]
    async fn delegate_allows_narrower_nonagentic_target() {
        // Option C: a non-agentic target on a DISTINCT, NARROWER profile
        // (caller `broad` = [git, cargo]; target `narrow` = [git]) is no
        // longer refused. It is dispatched and runs under its OWN (narrower)
        // policy — a non-agentic delegate has no tool registry to inherit, so
        // there is no escalation surface.
        let config = config_with_narrowed_target();
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, caller_policy)
            .with_root_config(config.clone())
            .with_caller_alias("caller");

        let resolved = tool
            .policy_for_target("target")
            .expect("a narrower non-agentic cross-profile target must be allowed");
        // The dispatched policy is the TARGET's own (least-privilege), not
        // the caller's broader profile.
        assert_eq!(
            resolved.risk_profile_name, "narrow",
            "delegate must run under the target's narrow policy, not the caller's"
        );
    }

    // ── Per-delegate provider-endpoint resolution (T7 misroute regression) ──
    //
    // Before the fix, every delegate reused the single inherited
    // `provider_runtime_options` (the root config's FIRST model provider), so an
    // ollama delegate posted to whatever endpoint the first provider declared. On
    // a host where a Z.AI alias sorts first with an explicit `uri`, ollama
    // delegates POSTed the local model to Z.AI and got `400 {"code":"1211"}`.
    // These tests pin the per-delegate-alias re-resolution.
    // (The magks originals also asserted the pre-fix leak via
    // `provider_runtime_options_from_config`, which beta-2 removed; those sanity
    // blocks are dropped here — the real regression assertions stand alone.)

    /// newmoon-shape config: a `zai` alias (explicit z.ai uri) sorts before
    /// `ollama` in the family-slot order, so it is `first_model_provider()`.
    const NEWMOON_SHAPE_TOML: &str = r#"
[providers.models.zai.orch]
uri = "https://api.z.ai/api/coding/paas/v4"
model = "glm-5.1"

[providers.models.ollama.local]
uri = "http://127.0.0.1:11434/v1"
model = "qwen3.5:9b"

[agents.worker]
model_provider = "ollama.local"

[agents.zai_worker]
model_provider = "zai.orch"
"#;

    /// arbot-shape config: NO provider sets an explicit `uri`; the earliest
    /// non-empty family (`anthropic`) is `first_model_provider()` with no uri.
    const ARBOT_SHAPE_TOML: &str = r#"
[providers.models.anthropic.default]
model = "claude-opus-4-20250514"

[providers.models.zai.comment]
model = "glm-5-turbo"

[providers.models.ollama.tag]
model = "qwen3.5:9b"

[agents.tagger]
model_provider = "ollama.tag"

[agents.commenter]
model_provider = "zai.comment"
"#;

    #[test]
    fn delegate_ollama_resolves_own_endpoint_not_first_provider_url() {
        let config: Config = toml::from_str(NEWMOON_SHAPE_TOML).unwrap();

        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_root_config(Arc::new(config));

        // The fix: the ollama delegate targets the ollama endpoint, NOT z.ai.
        assert_eq!(
            tool.resolve_delegate_provider_options("ollama.local")
                .provider_api_url
                .as_deref(),
            Some("http://127.0.0.1:11434/v1"),
            "ollama delegate must not inherit the first provider's z.ai url"
        );

        // A zai delegate still keeps its own (coding-plan, billing-safe) uri.
        assert_eq!(
            tool.resolve_delegate_provider_options("zai.orch")
                .provider_api_url
                .as_deref(),
            Some("https://api.z.ai/api/coding/paas/v4"),
        );
    }

    #[test]
    fn delegate_no_uri_config_falls_through_to_family_default() {
        // arbot-shape: no explicit uris anywhere. Must NOT regress — the ollama
        // delegate keeps `provider_api_url == None` (factory localhost default),
        // the zai delegate resolves to the zai family default.
        let config: Config = toml::from_str(ARBOT_SHAPE_TOML).unwrap();

        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_root_config(Arc::new(config));

        // ollama's FamilyEndpoint::endpoint_uri() is None → stays None →
        // create_model_provider falls back to the localhost:11434/v1 default.
        assert_eq!(
            tool.resolve_delegate_provider_options("ollama.tag")
                .provider_api_url,
            None,
            "no-uri ollama delegate must stay None (factory default) — arbot parity"
        );
        // zai's endpoint default (Global) IS the coding-plan endpoint.
        assert_eq!(
            tool.resolve_delegate_provider_options("zai.comment")
                .provider_api_url
                .as_deref(),
            Some("https://api.z.ai/api/coding/paas/v4"),
        );
    }

    #[test]
    fn delegate_options_fall_back_to_inherited_without_root_config() {
        // Legacy unit-test constructors / bare aliases must reuse the inherited
        // options unchanged (no root_config to re-resolve against).
        let inherited = zeroclaw_providers::ModelProviderRuntimeOptions {
            provider_api_url: Some("http://inherited.example/v1".to_string()),
            ..Default::default()
        };
        let tool = DelegateTool::new_with_options(
            HashMap::new(),
            None,
            test_security(),
            inherited.clone(),
        );

        assert_eq!(
            tool.resolve_delegate_provider_options("ollama.local")
                .provider_api_url,
            inherited.provider_api_url,
            "no root_config → inherited options"
        );
        assert_eq!(
            tool.resolve_delegate_provider_options("ollama")
                .provider_api_url,
            inherited.provider_api_url,
            "bare alias (no family.alias) → inherited options"
        );
    }

    // ── Option C: cross-profile narrowing gate ──────────────────────────────

    /// caller `broad` ([git, cargo], delegation allow) → target `narrow`
    /// ([git]) on an AGENTIC runtime profile. Narrower on commands, but the
    /// agentic loop would reuse the caller's tool registry, so it must refuse.
    fn config_narrower_agentic_target() -> Arc<zeroclaw_config::schema::Config> {
        use zeroclaw_config::autonomy::{DelegationMode, DelegationPolicy};
        use zeroclaw_config::schema::{
            AliasedAgentConfig, Config, RiskProfileConfig, RuntimeProfileConfig,
        };
        let mut config = Config::default();
        config.risk_profiles.insert(
            "broad".to_string(),
            RiskProfileConfig {
                allowed_commands: vec!["git".into(), "cargo".into()],
                delegation_policy: DelegationPolicy {
                    mode: DelegationMode::Allow,
                },
                ..RiskProfileConfig::default()
            },
        );
        config.risk_profiles.insert(
            "narrow".to_string(),
            RiskProfileConfig {
                allowed_commands: vec!["git".into()],
                allowed_tools: vec!["shell".into()],
                ..RiskProfileConfig::default()
            },
        );
        config.runtime_profiles.insert(
            "agentic_narrow".to_string(),
            RuntimeProfileConfig {
                agentic: true,
                ..RuntimeProfileConfig::default()
            },
        );
        config.agents.insert(
            "caller".to_string(),
            AliasedAgentConfig {
                risk_profile: "broad".to_string(),
                model_provider: "ollama.caller".into(),
                ..AliasedAgentConfig::default()
            },
        );
        config.agents.insert(
            "target".to_string(),
            AliasedAgentConfig {
                risk_profile: "narrow".to_string(),
                runtime_profile: "agentic_narrow".to_string(),
                model_provider: "ollama.target".into(),
                ..AliasedAgentConfig::default()
            },
        );
        Arc::new(config)
    }

    /// caller `restricted` (allowed_tools=[read_file], delegation allow) →
    /// target `wider_tools` (allowed_tools=[read_file, shell]). Same
    /// commands/caps; ONLY the tool allowlist broadens. Both non-agentic.
    fn config_tool_broadening() -> Arc<zeroclaw_config::schema::Config> {
        use zeroclaw_config::autonomy::{DelegationMode, DelegationPolicy};
        use zeroclaw_config::schema::{AliasedAgentConfig, Config, RiskProfileConfig};
        let mut config = Config::default();
        config.risk_profiles.insert(
            "restricted".to_string(),
            RiskProfileConfig {
                allowed_tools: vec!["read_file".into()],
                delegation_policy: DelegationPolicy {
                    mode: DelegationMode::Allow,
                },
                ..RiskProfileConfig::default()
            },
        );
        config.risk_profiles.insert(
            "wider_tools".to_string(),
            RiskProfileConfig {
                allowed_tools: vec!["read_file".into(), "shell".into()],
                ..RiskProfileConfig::default()
            },
        );
        config.agents.insert(
            "caller".to_string(),
            AliasedAgentConfig {
                risk_profile: "restricted".to_string(),
                model_provider: "ollama.caller".into(),
                ..AliasedAgentConfig::default()
            },
        );
        config.agents.insert(
            "target".to_string(),
            AliasedAgentConfig {
                risk_profile: "wider_tools".to_string(),
                model_provider: "ollama.target".into(),
                ..AliasedAgentConfig::default()
            },
        );
        Arc::new(config)
    }

    #[tokio::test]
    async fn delegate_refuses_agentic_cross_profile() {
        // An AGENTIC target on a different (even narrower) profile is refused:
        // the in-process agentic loop reuses the caller's tool registry, so it
        // cannot safely run under the target's policy. Narrowed agentic runs
        // go through spawn_subagent / a shared profile.
        let config = config_narrower_agentic_target();
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, caller_policy)
            .with_root_config(config.clone())
            .with_caller_alias("caller");

        let err = tool
            .policy_for_target("target")
            .expect_err("agentic cross-profile target must be refused");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("AGENTIC") && chain.contains("spawn_subagent"),
            "expected agentic-cross-profile refusal, got: {chain}"
        );
    }

    #[tokio::test]
    async fn delegate_refuses_tool_allowlist_broadening() {
        // ensure_no_escalation_beyond does not cover the tool allowlist; a
        // target whose explicit allowed_tools names a tool the caller cannot
        // use (shell) is a tool-authorization escalation and must be refused.
        let config = config_tool_broadening();
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, caller_policy)
            .with_root_config(config.clone())
            .with_caller_alias("caller");

        let err = tool
            .policy_for_target("target")
            .expect_err("a target whose tool allowlist exceeds the caller's must be refused");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("narrowing-only"),
            "expected tool-broadening refusal, got: {chain}"
        );
    }

    #[tokio::test]
    async fn delegate_forbidden_policy_blocks_before_narrowing() {
        // Gate 1 (operator on/off switch) is preserved: a caller whose
        // delegation_policy forbids delegation is refused even for an
        // otherwise-narrower target. config_with_narrowed_target's `broad`
        // profile sets allow; flip the caller's resolved policy to forbidden.
        let config = config_with_narrowed_target();
        let mut caller = SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves");
        caller.delegation_policy = zeroclaw_config::autonomy::DelegationPolicy {
            mode: zeroclaw_config::autonomy::DelegationMode::Forbidden,
        };
        let caller_policy = Arc::new(caller);
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, caller_policy)
            .with_root_config(config.clone())
            .with_caller_alias("caller");

        let err = tool
            .policy_for_target("target")
            .expect_err("forbidden delegation_policy must block");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("forbidden by the caller's delegation_policy"),
            "expected delegation_policy refusal, got: {chain}"
        );
    }

    #[test]
    fn parameters_schema_advertises_narrower_nonagentic_target() {
        // The roster advertisement mirrors the gate: a narrower non-agentic
        // target IS reachable and must be advertised.
        let config = config_with_narrowed_target();
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, caller_policy)
            .with_root_config(config.clone())
            .with_caller_alias("caller");
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap_or_default();
        assert!(
            desc.contains("target"),
            "narrower non-agentic target must be advertised: {desc}"
        );
    }

    #[test]
    fn parameters_schema_hides_broader_target() {
        // A broader cross-profile target is unreachable and must NOT be
        // advertised (so the orchestrator never proposes an escalating call).
        let config = config_with_two_agents("caller", 5, "target", 50);
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut delegate_agents = HashMap::new();
        for (name, agent) in &config.agents {
            delegate_agents.insert(name.clone(), agent.clone());
        }
        let tool = DelegateTool::new(delegate_agents, None, caller_policy)
            .with_root_config(config.clone())
            .with_caller_alias("caller");
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap_or_default();
        assert!(
            !desc.contains("target"),
            "broader target must NOT be advertised: {desc}"
        );
    }

    // ── HARDEN: comparison-completeness gaps closed in cross_profile_decision ──

    /// Build a 2-agent cross-profile config (caller `c_rp`, delegation
    /// allowed; target `t_rp`, optionally mutated) and return the
    /// narrowing-gate result for delegating caller → target. Shared by the
    /// HARDEN gap tests so each can vary exactly ONE capability dimension.
    fn harden_gate(
        caller_rp: RiskProfileConfig,
        target_rp: RiskProfileConfig,
        mutate_target: impl FnOnce(&mut AliasedAgentConfig),
    ) -> anyhow::Result<Arc<SecurityPolicy>> {
        use zeroclaw_config::autonomy::{DelegationMode, DelegationPolicy};
        let mut config = Config::default();
        let mut caller_rp = caller_rp;
        // The caller must permit delegation, else gate 1 rejects before the
        // narrowing comparison under test is reached.
        caller_rp.delegation_policy = DelegationPolicy {
            mode: DelegationMode::Allow,
        };
        config.risk_profiles.insert("c_rp".to_string(), caller_rp);
        config.risk_profiles.insert("t_rp".to_string(), target_rp);
        config.agents.insert(
            "caller".to_string(),
            AliasedAgentConfig {
                risk_profile: "c_rp".to_string(),
                model_provider: "ollama.caller".into(),
                ..AliasedAgentConfig::default()
            },
        );
        let mut target = AliasedAgentConfig {
            risk_profile: "t_rp".to_string(),
            model_provider: "ollama.target".into(),
            ..AliasedAgentConfig::default()
        };
        mutate_target(&mut target);
        config.agents.insert("target".to_string(), target);
        let config = Arc::new(config);
        let caller_policy =
            Arc::new(SecurityPolicy::for_agent(&config, "caller").expect("caller policy resolves"));
        let mut agents = HashMap::new();
        for (name, agent) in &config.agents {
            agents.insert(name.clone(), agent.clone());
        }
        DelegateTool::new(agents, None, caller_policy)
            .with_root_config(config)
            .with_caller_alias("caller")
            .policy_for_target("target")
    }

    /// GAP 1 [HIGH] — workspace FS tiers. `profile_grants` compares grants
    /// built on a shared neutral base with the cross-agent FS tiers
    /// (`workspace.access`, `unrestricted_filesystem` → `workspace_only`)
    /// re-applied. A target broader on any of those tiers is now refused; a
    /// same-scope target (whose only difference is its per-agent workspace
    /// jail, relative OR absolute) is still allowed.
    #[tokio::test]
    async fn delegate_refuses_broader_workspace_fs_target() {
        use zeroclaw_config::multi_agent::{AccessMode, AgentAlias};

        // (a) `unrestricted_filesystem` clears `workspace_only` — strictly
        // broader filesystem reach. Every from_profiles dimension is identical,
        // so the pre-HARDEN gate (which never read this flag) ALLOWED it.
        let err = harden_gate(
            RiskProfileConfig::default(),
            RiskProfileConfig::default(),
            |t| t.workspace.unrestricted_filesystem = true,
        )
        .expect_err("a target with unrestricted_filesystem is broader and must be refused");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("narrowing-only"),
            "unrestricted_filesystem must be refused as narrowing violation: {chain}"
        );

        // (b) a `workspace.access` grant adds a sibling workspace root the
        // caller lacks — a cross-agent read+write tier the old from_profiles
        // build hard-coded to empty and never compared.
        let err = harden_gate(
            RiskProfileConfig::default(),
            RiskProfileConfig::default(),
            |t| {
                t.workspace
                    .access
                    .insert(AgentAlias::new("sib"), AccessMode::ReadWrite);
            },
        )
        .expect_err("a target granted a sibling workspace root must be refused");
        let chain = format!("{err:#}");
        assert!(
            chain.contains("narrowing-only"),
            "workspace.access broadening must be refused: {chain}"
        );

        // (c) NO false refusal: two cross-profile agents whose risk profiles
        // each declare the SAME workspace-RELATIVE allowed_root. Resolving both
        // on the shared neutral base aligns them, so the narrower-equal target
        // is correctly ALLOWED.
        let rel_root = || RiskProfileConfig {
            allowed_roots: vec!["data".into()],
            ..RiskProfileConfig::default()
        };
        let resolved = harden_gate(rel_root(), rel_root(), |_| {})
            .expect("workspace-relative roots must align on the shared base (no false refusal)");
        assert_eq!(
            resolved.risk_profile_name, "t_rp",
            "a same-scope cross-profile target must run under its own policy"
        );

        // (d) NO false refusal (regression for the neutralization rewrite): a
        // SHARED ABSOLUTE risk-profile root that NESTS under the target's
        // custom `workspace.path` must still compare equal to the caller's
        // identical literal root. The earlier per-agent neutralization rewrote
        // the root onto the sentinel for the nesting agent only, false-refusing
        // provably-identical grants; resolving on a shared base (never the
        // per-agent workspace) keeps the absolute root literal on BOTH sides.
        let abs_root = || RiskProfileConfig {
            allowed_roots: vec!["/srv/proj/src".into()],
            ..RiskProfileConfig::default()
        };
        let resolved = harden_gate(abs_root(), abs_root(), |t| {
            t.workspace.path = Some(std::path::PathBuf::from("/srv/proj"));
        })
        .expect("identical absolute roots must compare equal even when a workspace.path nests them");
        assert_eq!(
            resolved.risk_profile_name, "t_rp",
            "shared absolute roots must not false-refuse under a custom workspace.path"
        );
    }

    /// GAP 2 [MED] — tool / approval / sandbox dimensions. `ensure_no_escalation_beyond`
    /// covers none of these; `target_tools_within_caller` (now both
    /// allowed_tools arms + excluded_tools) and `target_approval_sandbox_within_caller`
    /// close them. Each sub-case varies exactly ONE dimension to broaden the
    /// target; one negative control proves no false refusal.
    #[tokio::test]
    async fn delegate_refuses_relaxed_tool_approval_sandbox_dims() {
        let base = RiskProfileConfig::default;

        // (a) excluded_tools: caller denies "shell"; target denies nothing →
        // the target re-authorizes a tool the caller excludes. REFUSED.
        let mut caller = base();
        caller.excluded_tools = vec!["shell".into()];
        let err = harden_gate(caller.clone(), base(), |_| {}).expect_err(
            "a smaller target excluded_tools set re-authorizes a caller-denied tool",
        );
        assert!(
            format!("{err:#}").contains("narrowing-only"),
            "excluded_tools broadening: {err:#}"
        );

        // ...and a target that ALSO excludes shell (a superset denylist) is
        // narrower → ALLOWED (no false refusal).
        let mut target = base();
        target.excluded_tools = vec!["shell".into(), "file_write".into()];
        let ok = harden_gate(caller, target, |_| {});
        assert!(
            ok.is_ok(),
            "a target with a superset excluded_tools set must be allowed: {ok:?}"
        );

        // (b) allowed_tools (Some, None) arm: caller restricts to [read_file];
        // the target carries no allowlist (unrestricted) → REFUSED. The
        // pre-HARDEN (Some, Some)-only check skipped this arm.
        let mut caller = base();
        caller.allowed_tools = vec!["read_file".into()];
        let err = harden_gate(caller, base(), |_| {}).expect_err(
            "an unrestricted target under a caller that restricts allowed_tools must be refused",
        );
        assert!(
            format!("{err:#}").contains("narrowing-only"),
            "(Some,None) allowed_tools arm: {err:#}"
        );

        // (c) auto_approve: the target auto-approves a tool the caller does
        // not — bypassing an approval the caller requires. REFUSED.
        let mut target = base();
        target.auto_approve.push("shell".into());
        let err = harden_gate(base(), target, |_| {})
            .expect_err("a target auto-approving a tool the caller does not must be refused");
        assert!(
            format!("{err:#}").contains("auto-approves"),
            "auto_approve broadening: {err:#}"
        );

        // (d) always_ask: the caller requires always-ask for file_write; the
        // target drops it. REFUSED.
        let mut caller = base();
        caller.always_ask = vec!["file_write".into()];
        let err = harden_gate(caller, base(), |_| {})
            .expect_err("a target dropping an always_ask the caller requires must be refused");
        assert!(
            format!("{err:#}").contains("always-ask"),
            "always_ask drop: {err:#}"
        );

        // (e) sandbox: the caller runs sandboxed via the COMMON active-by-
        // default regime (backend set, `sandbox_enabled` unset → None →
        // effectively sandboxed); the target sets `sandbox_enabled = false`
        // → NoopSandbox, strictly unsandboxed. REFUSED. The pre-fix
        // `caller.sandbox_enabled == Some(true)` guard MISSED this (the
        // caller's flag is None), which the adversarial review caught.
        let mut caller = base();
        caller.sandbox_backend = Some("firejail".into());
        let mut target = base();
        target.sandbox_backend = Some("firejail".into());
        target.sandbox_enabled = Some(false);
        let err = harden_gate(caller, target, |_| {})
            .expect_err("a target disabling the sandbox under an active-by-default caller must be refused");
        assert!(
            format!("{err:#}").contains("unsandboxed"),
            "sandbox downgrade (None-but-active caller): {err:#}"
        );

        // (e2) sandbox backend `none` → NoopSandbox even with enabled unset →
        // strictly unsandboxed under a sandboxed caller. REFUSED.
        let mut caller = base();
        caller.sandbox_backend = Some("firejail".into());
        let mut target = base();
        target.sandbox_backend = Some("none".into());
        let err = harden_gate(caller, target, |_| {})
            .expect_err("a target with sandbox_backend=none under a sandboxed caller must be refused");
        assert!(
            format!("{err:#}").contains("unsandboxed"),
            "sandbox backend=none: {err:#}"
        );

        // (e3) NO false refusal: `firejail_args` is runtime-inert (the runtime
        // hard-codes the firejail flag set and never forwards policy
        // firejail_args), so a target with DIFFERENT firejail_args produces a
        // byte-identical sandbox and must NOT be refused. Deliberate
        // non-comparison; becomes an equality check once the runtime forwards
        // firejail_args (the arg space is non-monotone). Guards against the
        // over-strict equality clause the adversarial re-verify flagged.
        let mut caller = base();
        caller.sandbox_backend = Some("firejail".into());
        caller.firejail_args = vec!["--net=none".into()];
        let mut target = base();
        target.sandbox_backend = Some("firejail".into());
        target.firejail_args = vec!["--net=none".into(), "--noprofile".into()];
        let ok = harden_gate(caller, target, |_| {});
        assert!(
            ok.is_ok(),
            "differing but runtime-inert firejail_args must not false-refuse: {ok:?}"
        );

        // (e4) NO false refusal: identical sandbox config (same backend, both
        // active-by-default) is narrower-or-equal → ALLOWED.
        let mut caller = base();
        caller.sandbox_backend = Some("firejail".into());
        let mut target = base();
        target.sandbox_backend = Some("firejail".into());
        let ok = harden_gate(caller, target, |_| {});
        assert!(
            ok.is_ok(),
            "identical sandbox config must be allowed (no false refusal): {ok:?}"
        );
    }

    /// GAP 3 [LOW] — agentic-lookup trim divergence. `cross_profile_decision`
    /// reads `agentic` via a TRIMMED `runtime_profiles.get(...)`; dispatch's
    /// `resolve_agentic` previously did an UNTRIMMED get, so a whitespace-
    /// padded `runtime_profile` could make the gate read non-agentic while
    /// dispatch read agentic (or vice versa). `resolve_agentic` now trims, so
    /// the two can never disagree.
    #[test]
    fn resolve_agentic_trims_runtime_profile_reference() {
        let mut runtime_profiles = HashMap::new();
        runtime_profiles.insert(
            "agentic_rt".to_string(),
            RuntimeProfileConfig {
                agentic: true,
                ..RuntimeProfileConfig::default()
            },
        );
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_runtime_profiles(runtime_profiles);

        // The trimmed lookup matches the gate's `runtime_profile.trim()` read;
        // an untrimmed get would miss the padded reference and read false.
        assert!(
            tool.resolve_agentic("  agentic_rt  "),
            "resolve_agentic must trim the reference to match the gate"
        );
        assert!(tool.resolve_agentic("agentic_rt"));
        assert!(!tool.resolve_agentic("   "));
        assert!(!tool.resolve_agentic("unknown_rt"));
    }
}
