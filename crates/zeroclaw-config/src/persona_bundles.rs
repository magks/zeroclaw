//! Persona-bundle directory rules and helpers.
//!
//! Single source of truth for:
//! - the `shared/personas/<alias>/` default directory
//! - the per-config uniqueness rule
//! - the existence / is-directory check
//!
//! Lives in `zeroclaw-config` (not `zeroclaw-runtime`) so
//! [`crate::schema::Config::validate`] can call into it at load time and the
//! runtime personality loader can reuse the same resolution; there is no
//! second implementation.
//!
//! **How this differs from [`crate::skill_bundles`].** Skill bundles are
//! executable capability and are auto-provisioned, so their rule is
//! containment inside `<install>/shared/` (a sandbox boundary) and the
//! directory is created on demand. Persona bundles are *authored content* —
//! identity markdown a user keeps in a dedicated library dir, often outside
//! the install — and they are trusted config. The security boundary for what
//! a persona may *read at runtime* is the policy layer
//! (`risk_profiles.<X>.allowed_roots` plus a bundle's own
//! `extra_allowed_roots`), not the bundle directory's location. So persona
//! bundles may resolve to any absolute path, and instead of containment we
//! validate that the directory actually exists and is a directory — a missing
//! path is almost always a typo, and (unlike skills) nothing auto-creates it,
//! so an unflagged miss would silently overlay nothing.

use std::path::{Path, PathBuf};

use crate::schema::Config;

/// Canonical default directory for a persona bundle:
/// `<install>/shared/personas/<alias>/`.
#[must_use]
pub fn default_directory(install_root: &Path, alias: &str) -> PathBuf {
    install_root.join("shared").join("personas").join(alias)
}

/// Resolve the on-disk directory for a configured persona bundle, applying the
/// default when `[persona-bundles.<alias>].directory` is unset or empty.
/// Absolute paths configured by the operator pass through verbatim; relative
/// paths are resolved against the install root.
pub fn resolve_directory(
    config: &Config,
    install_root: &Path,
    alias: &str,
) -> Result<PathBuf, PersonaBundleDirectoryError> {
    let bundle = config
        .persona_bundles
        .get(alias)
        .ok_or_else(|| PersonaBundleDirectoryError::UnknownBundle(alias.to_string()))?;

    let configured = bundle
        .directory
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let path = match configured {
        Some(raw) => {
            let candidate = PathBuf::from(raw);
            if candidate.is_absolute() {
                candidate
            } else {
                install_root.join(candidate)
            }
        }
        None => default_directory(install_root, alias),
    };
    Ok(path)
}

/// Reject a configured persona-bundle directory that does not exist or is not
/// a directory. Unlike [`crate::skill_bundles::validate_directory`] there is no
/// `<install>/shared/` containment rule (see the module docs); persona bundles
/// are trusted, location-free authored content. Run inside
/// [`crate::schema::Config::validate`], where failures are demoted to startup
/// warnings — surfacing a typo'd path without blocking boot.
pub fn validate_directory(path: &Path) -> Result<(), PersonaBundleDirectoryError> {
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_dir() => Ok(()),
        Ok(_) => Err(PersonaBundleDirectoryError::NotADirectory {
            path: path.display().to_string(),
        }),
        Err(_) => Err(PersonaBundleDirectoryError::Missing {
            path: path.display().to_string(),
        }),
    }
}

/// Reject configs where two persona bundles resolve to the same directory.
pub fn validate_uniqueness(
    config: &Config,
    install_root: &Path,
) -> Result<(), PersonaBundleDirectoryError> {
    let mut seen: Vec<(String, PathBuf)> = Vec::with_capacity(config.persona_bundles.len());
    for alias in config.persona_bundles.keys() {
        let dir = resolve_directory(config, install_root, alias)?;
        let normalized = crate::paths::normalize_lexical(&dir);
        if let Some((other, _)) = seen.iter().find(|(_, p)| p == &normalized) {
            return Err(PersonaBundleDirectoryError::DirectoryCollision {
                path: normalized.display().to_string(),
                first: other.clone(),
                second: alias.clone(),
            });
        }
        seen.push((alias.clone(), normalized));
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum PersonaBundleDirectoryError {
    #[error("persona bundle '{0}' is not configured")]
    UnknownBundle(String),

    #[error("persona-bundle directory '{path}' does not exist")]
    Missing { path: String },

    #[error("persona-bundle directory '{path}' is not a directory")]
    NotADirectory { path: String },

    #[error(
        "persona-bundles '{first}' and '{second}' both resolve to directory '{path}'; each bundle must own a unique directory"
    )]
    DirectoryCollision {
        path: String,
        first: String,
        second: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::PersonaBundleConfig;

    fn cfg_with_bundle(alias: &str, directory: Option<&str>) -> Config {
        let mut cfg = Config::default();
        cfg.persona_bundles.insert(
            alias.to_string(),
            PersonaBundleConfig {
                directory: directory.map(String::from),
                ..Default::default()
            },
        );
        cfg
    }

    #[test]
    fn defaults_to_shared_personas_alias_when_unset() {
        let cfg = cfg_with_bundle("alpha", None);
        let root = Path::new("/tmp/install");
        let resolved = resolve_directory(&cfg, root, "alpha").unwrap();
        assert_eq!(resolved, root.join("shared/personas/alpha"));
    }

    #[test]
    fn empty_directory_string_is_treated_as_unset() {
        let cfg = cfg_with_bundle("alpha", Some("   "));
        let root = Path::new("/tmp/install");
        assert_eq!(
            resolve_directory(&cfg, root, "alpha").unwrap(),
            root.join("shared/personas/alpha"),
        );
    }

    #[test]
    fn absolute_directory_passes_through_unchanged() {
        // Personas legitimately live outside <install>/shared/ (e.g. a
        // dedicated library dir); an absolute path must not be rewritten.
        let cfg = cfg_with_bundle("alpha", Some("/home/u/personas/larry-r-v0"));
        let root = Path::new("/tmp/install");
        assert_eq!(
            resolve_directory(&cfg, root, "alpha").unwrap(),
            PathBuf::from("/home/u/personas/larry-r-v0"),
        );
    }

    #[test]
    fn relative_directory_resolves_against_install_root() {
        let cfg = cfg_with_bundle("alpha", Some("personas/alpha"));
        let root = Path::new("/tmp/install");
        assert_eq!(
            resolve_directory(&cfg, root, "alpha").unwrap(),
            root.join("personas/alpha"),
        );
    }

    #[test]
    fn resolve_unknown_bundle_errors() {
        let cfg = Config::default();
        let err = resolve_directory(&cfg, Path::new("/tmp/install"), "ghost").unwrap_err();
        assert!(matches!(err, PersonaBundleDirectoryError::UnknownBundle(_)));
    }

    #[test]
    fn validate_directory_accepts_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(validate_directory(dir.path()).is_ok());
    }

    #[test]
    fn validate_directory_rejects_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        let err = validate_directory(&missing).unwrap_err();
        assert!(matches!(err, PersonaBundleDirectoryError::Missing { .. }));
    }

    #[test]
    fn validate_directory_rejects_a_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let err = validate_directory(file.path()).unwrap_err();
        assert!(matches!(
            err,
            PersonaBundleDirectoryError::NotADirectory { .. }
        ));
    }

    #[test]
    fn uniqueness_rejects_two_bundles_pointing_at_same_dir() {
        let mut cfg = Config::default();
        cfg.persona_bundles.insert(
            "alpha".into(),
            PersonaBundleConfig {
                directory: Some("personas/shared-pool".into()),
                ..Default::default()
            },
        );
        cfg.persona_bundles.insert(
            "beta".into(),
            PersonaBundleConfig {
                directory: Some("personas/shared-pool".into()),
                ..Default::default()
            },
        );
        let err = validate_uniqueness(&cfg, Path::new("/tmp/install")).unwrap_err();
        assert!(matches!(
            err,
            PersonaBundleDirectoryError::DirectoryCollision { .. }
        ));
    }

    #[test]
    fn uniqueness_passes_for_distinct_default_directories() {
        let mut cfg = Config::default();
        cfg.persona_bundles
            .insert("alpha".into(), PersonaBundleConfig::default());
        cfg.persona_bundles
            .insert("beta".into(), PersonaBundleConfig::default());
        validate_uniqueness(&cfg, Path::new("/tmp/install")).unwrap();
    }
}
