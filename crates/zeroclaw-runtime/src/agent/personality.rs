//! Personality system — loads workspace identity files (SOUL.md, IDENTITY.md,
//! USER.md) and injects them into the system prompt pipeline.
//!
//! Ported from RustyClaw `src/agent/personality.rs`.  The loader reads markdown
//! files from the workspace root, validates size limits, and produces a
//! [`PersonalityProfile`] that the prompt builder can render.

use std::fmt::Write;
use std::path::{Path, PathBuf};

/// Maximum characters per personality file before truncation.
pub const MAX_FILE_CHARS: usize = 20_000;

/// Well-known personality files loaded from the workspace root.
pub const PERSONALITY_FILES: &[&str] = &[
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "AGENTS.md",
    "TOOLS.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
];

/// Subset of [`PERSONALITY_FILES`] that the dashboard exposes for
/// authoring. `BOOTSTRAP.md` is deliberately excluded: it's a
/// first-run scaffold the agent reads once and deletes, not a file
/// the user is meant to hand-edit. The runtime still injects it when
/// it exists on disk.
pub const EDITABLE_PERSONALITY_FILES: &[&str] = &[
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "AGENTS.md",
    "TOOLS.md",
    "HEARTBEAT.md",
    "MEMORY.md",
];

/// A single personality file loaded from the workspace.
#[derive(Debug, Clone)]
pub struct PersonalityFile {
    /// Filename (e.g. `SOUL.md`).
    pub name: String,
    /// Raw content (possibly truncated).
    pub content: String,
    /// Whether the content was truncated due to size limits.
    pub truncated: bool,
    /// Full path on disk.
    pub path: PathBuf,
}

/// Aggregated personality profile loaded from a workspace.
#[derive(Debug, Clone, Default)]
pub struct PersonalityProfile {
    /// Successfully loaded personality files.
    pub files: Vec<PersonalityFile>,
    /// Files that were expected but not found.
    pub missing: Vec<String>,
}

impl PersonalityProfile {
    /// Returns the content of a specific file by name, if loaded.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.files
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.content.as_str())
    }

    /// Returns `true` if no personality files were loaded.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Render all loaded personality files into a prompt fragment.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for file in &self.files {
            let _ = writeln!(out, "### {}\n", file.name);
            out.push_str(&file.content);
            if file.truncated {
                let _ = writeln!(
                    out,
                    "\n\n[... truncated at {MAX_FILE_CHARS} chars — use `read` for full file]\n"
                );
            } else {
                out.push_str("\n\n");
            }
        }
        out
    }
}

/// A persona bundle resolved to an absolute directory, ready for the overlay
/// loader. Built at agent-build time from a `[persona_bundles.<alias>]` entry
/// via [`ResolvedPersonaBundle::new`], which is also where the bundle's
/// declared format is validated — so by the time a bundle reaches the loader
/// it is guaranteed to be a loadable (markdown / `openclaw`) bundle.
#[derive(Debug, Clone)]
pub struct ResolvedPersonaBundle {
    /// Absolute directory the bundle's personality files live in.
    pub directory: PathBuf,
    /// File names to include; empty means every recognised personality file.
    pub include: Vec<String>,
    /// File names to exclude from this bundle.
    pub exclude: Vec<String>,
}

impl ResolvedPersonaBundle {
    /// Build a resolved bundle, validating the declared `format`. Only the
    /// `openclaw` (markdown) format is loadable today; `aieos` — and any other
    /// value — is rejected with a clear error until AIEOS overlay support
    /// lands. An empty `format` is treated as the `openclaw` default (matching
    /// `PersonaBundleConfig`'s default).
    pub fn new(
        directory: PathBuf,
        format: &str,
        include: Vec<String>,
        exclude: Vec<String>,
    ) -> Result<Self, PersonaLoadError> {
        match format {
            "" | "openclaw" => Ok(Self {
                directory,
                include,
                exclude,
            }),
            other => Err(PersonaLoadError::UnsupportedFormat {
                format: other.to_string(),
                directory: directory.display().to_string(),
            }),
        }
    }

    /// Whether this bundle contributes `filename`, honouring include/exclude.
    /// `exclude` wins over `include`; an empty `include` admits every file.
    fn admits(&self, filename: &str) -> bool {
        if self.exclude.iter().any(|e| e == filename) {
            return false;
        }
        self.include.is_empty() || self.include.iter().any(|i| i == filename)
    }
}

/// Error building a [`ResolvedPersonaBundle`] from config.
#[derive(Debug, thiserror::Error)]
pub enum PersonaLoadError {
    #[error(
        "persona bundle at '{directory}' uses unsupported format '{format}'; only 'openclaw' (markdown) bundles are supported"
    )]
    UnsupportedFormat { format: String, directory: String },
}

/// Loads personality files from a workspace directory.
///
/// Each well-known file is read and validated.  Missing files are recorded
/// in `PersonalityProfile::missing` rather than treated as errors.
pub fn load_personality(workspace_dir: &Path) -> PersonalityProfile {
    load_personality_files(workspace_dir, PERSONALITY_FILES)
}

/// Load a specific set of personality files from a workspace directory.
pub fn load_personality_files(workspace_dir: &Path, filenames: &[&str]) -> PersonalityProfile {
    let mut profile = PersonalityProfile::default();
    for &filename in filenames {
        match read_personality_file(workspace_dir, filename) {
            Some(file) => profile.files.push(file),
            None => profile.missing.push(filename.to_string()),
        }
    }
    profile
}

/// Load the well-known personality files with persona-bundle overlay.
///
/// Resolution precedence for each file, highest first: the agent's own
/// `workspace_dir` (its per-instance copy-on-write layer), then each bundle
/// latest-listed first (so later bundles overlay earlier ones), then earlier
/// bundles. A file that is absent or effectively empty at a layer counts as
/// "not provided" there, so a lower layer can still supply it.
///
/// With an empty `bundles` slice this is identical to [`load_personality`].
pub fn load_personality_with_bundles(
    bundles: &[ResolvedPersonaBundle],
    workspace_dir: &Path,
) -> PersonalityProfile {
    let mut profile = PersonalityProfile::default();
    for &filename in PERSONALITY_FILES {
        match load_overlaid_file(filename, bundles, workspace_dir) {
            Some(file) => profile.files.push(file),
            None => profile.missing.push(filename.to_string()),
        }
    }
    profile
}

/// Resolve and read a single personality file under bundle-overlay precedence
/// (see [`load_personality_with_bundles`]). Returns the winning layer's file,
/// or `None` if no layer provides a non-empty copy. Exposed so the live-CLI
/// prompt path can share the exact same precedence as the ACP path.
pub fn load_overlaid_file(
    filename: &str,
    bundles: &[ResolvedPersonaBundle],
    workspace_dir: &Path,
) -> Option<PersonalityFile> {
    // Highest priority: the agent's own workspace (per-instance overrides).
    if let Some(file) = read_personality_file(workspace_dir, filename) {
        return Some(file);
    }
    // Then bundles, latest-listed first — later bundles overlay earlier ones.
    for bundle in bundles.iter().rev() {
        if !bundle.admits(filename) {
            continue;
        }
        if let Some(file) = read_personality_file(&bundle.directory, filename) {
            return Some(file);
        }
    }
    None
}

/// Read one personality file from `dir`, returning `None` if it is absent or
/// effectively empty (whitespace only). Applies the [`MAX_FILE_CHARS`] cap.
/// Single source of truth for the read+empty+truncate rule shared by every
/// loader entry point.
fn read_personality_file(dir: &Path, filename: &str) -> Option<PersonalityFile> {
    let path = dir.join(filename);
    let raw = std::fs::read_to_string(&path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (content, truncated) = truncate_content(trimmed);
    Some(PersonalityFile {
        name: filename.to_string(),
        content,
        truncated,
        path,
    })
}

/// Truncate content to `MAX_FILE_CHARS` if necessary.
fn truncate_content(content: &str) -> (String, bool) {
    if content.chars().count() <= MAX_FILE_CHARS {
        return (content.to_string(), false);
    }
    let truncated = content
        .char_indices()
        .nth(MAX_FILE_CHARS)
        .map(|(idx, _)| &content[..idx])
        .unwrap_or(content);
    (truncated.to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_workspace(files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zeroclaw_personality_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, content) in files {
            std::fs::write(dir.join(name), content).unwrap();
        }
        dir
    }

    #[test]
    fn load_personality_reads_existing_files() {
        let ws = setup_workspace(&[
            ("SOUL.md", "I am a helpful assistant."),
            ("IDENTITY.md", "Name: Nova"),
        ]);

        let profile = load_personality(&ws);
        assert_eq!(profile.files.len(), 2);
        assert_eq!(profile.get("SOUL.md").unwrap(), "I am a helpful assistant.");
        assert_eq!(profile.get("IDENTITY.md").unwrap(), "Name: Nova");
        assert!(!profile.is_empty());

        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn load_personality_records_missing_files() {
        let ws = setup_workspace(&[("SOUL.md", "soul content")]);

        let profile = load_personality(&ws);
        assert_eq!(profile.files.len(), 1);
        assert!(profile.missing.contains(&"IDENTITY.md".to_string()));
        assert!(profile.missing.contains(&"USER.md".to_string()));

        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn load_personality_treats_empty_files_as_missing() {
        let ws = setup_workspace(&[("SOUL.md", "   \n  ")]);

        let profile = load_personality(&ws);
        assert!(profile.is_empty());
        assert!(profile.missing.contains(&"SOUL.md".to_string()));

        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn load_personality_truncates_large_files() {
        let large = "x".repeat(MAX_FILE_CHARS + 500);
        let ws = setup_workspace(&[("SOUL.md", &large)]);

        let profile = load_personality(&ws);
        let soul = profile.files.iter().find(|f| f.name == "SOUL.md").unwrap();
        assert!(soul.truncated);
        assert_eq!(soul.content.chars().count(), MAX_FILE_CHARS);

        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn render_produces_markdown_sections() {
        let ws = setup_workspace(&[("SOUL.md", "Be kind."), ("IDENTITY.md", "Name: Nova")]);

        let profile = load_personality(&ws);
        let rendered = profile.render();
        assert!(rendered.contains("### SOUL.md"));
        assert!(rendered.contains("Be kind."));
        assert!(rendered.contains("### IDENTITY.md"));
        assert!(rendered.contains("Name: Nova"));

        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn render_truncated_file_shows_notice() {
        let large = "y".repeat(MAX_FILE_CHARS + 100);
        let ws = setup_workspace(&[("SOUL.md", &large)]);

        let profile = load_personality(&ws);
        let rendered = profile.render();
        assert!(rendered.contains("[... truncated at"));

        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn get_returns_none_for_missing_file() {
        let ws = setup_workspace(&[]);
        let profile = load_personality(&ws);
        assert!(profile.get("SOUL.md").is_none());
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn load_personality_files_custom_subset() {
        let ws = setup_workspace(&[("SOUL.md", "soul"), ("USER.md", "user")]);

        let profile = load_personality_files(&ws, &["SOUL.md", "USER.md"]);
        assert_eq!(profile.files.len(), 2);
        assert!(profile.missing.is_empty());

        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn empty_workspace_yields_empty_profile() {
        let ws = setup_workspace(&[]);
        let profile = load_personality(&ws);
        assert!(profile.is_empty());
        assert!(!profile.missing.is_empty());
        let _ = std::fs::remove_dir_all(ws);
    }

    // ── Persona-bundle overlay ───────────────────────────────────────

    fn bundle(dir: &Path) -> ResolvedPersonaBundle {
        ResolvedPersonaBundle::new(dir.to_path_buf(), "openclaw", vec![], vec![]).unwrap()
    }

    #[test]
    fn with_bundles_empty_matches_load_personality() {
        // Backwards-compat invariant: no bundles => identical to today's loader.
        let ws = setup_workspace(&[("SOUL.md", "soul"), ("USER.md", "user")]);
        let plain = load_personality(&ws);
        let overlaid = load_personality_with_bundles(&[], &ws);
        let pairs = |p: &PersonalityProfile| {
            p.files
                .iter()
                .map(|f| (f.name.clone(), f.content.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(pairs(&overlaid), pairs(&plain));
        assert_eq!(overlaid.missing, plain.missing);
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn workspace_overrides_bundles() {
        let b0 = setup_workspace(&[("SOUL.md", "bundle soul")]);
        let ws = setup_workspace(&[("SOUL.md", "workspace soul")]);
        let profile = load_personality_with_bundles(&[bundle(&b0)], &ws);
        assert_eq!(profile.get("SOUL.md").unwrap(), "workspace soul");
        let _ = std::fs::remove_dir_all(b0);
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn later_bundle_overlays_earlier() {
        let b0 = setup_workspace(&[("SOUL.md", "b0 soul"), ("USER.md", "b0 user")]);
        let b1 = setup_workspace(&[("SOUL.md", "b1 soul")]);
        let ws = setup_workspace(&[]);
        let profile = load_personality_with_bundles(&[bundle(&b0), bundle(&b1)], &ws);
        // b1 (later) wins SOUL; USER (only in b0) still shows through.
        assert_eq!(profile.get("SOUL.md").unwrap(), "b1 soul");
        assert_eq!(profile.get("USER.md").unwrap(), "b0 user");
        let _ = std::fs::remove_dir_all(b0);
        let _ = std::fs::remove_dir_all(b1);
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn bundle_fills_file_absent_from_workspace() {
        let b0 = setup_workspace(&[("IDENTITY.md", "bundle identity")]);
        let ws = setup_workspace(&[("SOUL.md", "ws soul")]);
        let profile = load_personality_with_bundles(&[bundle(&b0)], &ws);
        assert_eq!(profile.get("SOUL.md").unwrap(), "ws soul");
        assert_eq!(profile.get("IDENTITY.md").unwrap(), "bundle identity");
        let _ = std::fs::remove_dir_all(b0);
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn empty_workspace_file_falls_through_to_bundle() {
        // An empty (whitespace-only) workspace file must not blank out a
        // non-empty bundle file — empty counts as "not provided".
        let b0 = setup_workspace(&[("SOUL.md", "bundle soul")]);
        let ws = setup_workspace(&[("SOUL.md", "   \n ")]);
        let profile = load_personality_with_bundles(&[bundle(&b0)], &ws);
        assert_eq!(profile.get("SOUL.md").unwrap(), "bundle soul");
        let _ = std::fs::remove_dir_all(b0);
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn include_limits_bundle_contribution() {
        let b0 = setup_workspace(&[("SOUL.md", "b soul"), ("IDENTITY.md", "b identity")]);
        let ws = setup_workspace(&[]);
        let b = ResolvedPersonaBundle::new(b0.clone(), "openclaw", vec!["SOUL.md".into()], vec![])
            .unwrap();
        let profile = load_personality_with_bundles(&[b], &ws);
        assert_eq!(profile.get("SOUL.md").unwrap(), "b soul");
        assert!(profile.get("IDENTITY.md").is_none());
        assert!(profile.missing.contains(&"IDENTITY.md".to_string()));
        let _ = std::fs::remove_dir_all(b0);
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn exclude_removes_bundle_file() {
        let b0 = setup_workspace(&[("SOUL.md", "b soul"), ("IDENTITY.md", "b identity")]);
        let ws = setup_workspace(&[]);
        let b =
            ResolvedPersonaBundle::new(b0.clone(), "openclaw", vec![], vec!["IDENTITY.md".into()])
                .unwrap();
        let profile = load_personality_with_bundles(&[b], &ws);
        assert_eq!(profile.get("SOUL.md").unwrap(), "b soul");
        assert!(profile.get("IDENTITY.md").is_none());
        let _ = std::fs::remove_dir_all(b0);
        let _ = std::fs::remove_dir_all(ws);
    }

    #[test]
    fn resolved_bundle_rejects_aieos_format() {
        let err = ResolvedPersonaBundle::new(PathBuf::from("/tmp/x"), "aieos", vec![], vec![])
            .unwrap_err();
        assert!(matches!(err, PersonaLoadError::UnsupportedFormat { .. }));
    }

    #[test]
    fn resolved_bundle_accepts_openclaw_and_empty_format() {
        assert!(
            ResolvedPersonaBundle::new(PathBuf::from("/tmp/x"), "openclaw", vec![], vec![]).is_ok()
        );
        assert!(ResolvedPersonaBundle::new(PathBuf::from("/tmp/x"), "", vec![], vec![]).is_ok());
    }
}
