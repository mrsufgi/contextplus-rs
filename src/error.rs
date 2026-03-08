use thiserror::Error;

#[derive(Error, Debug)]
pub enum ContextPlusError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Ollama error: {0}")]
    Ollama(String),
    #[error("Tree-sitter parse error: {0}")]
    TreeSitter(String),
    #[error("Path traversal detected: {0}")]
    PathTraversal(String),
    #[error("Cache error: {0}")]
    Cache(String),
    #[error("Config error: {0}")]
    Config(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ContextPlusError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_converts() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: ContextPlusError = io_err.into();
        assert!(matches!(err, ContextPlusError::Io(_)));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn ollama_error_displays() {
        let err = ContextPlusError::Ollama("connection refused".into());
        assert_eq!(err.to_string(), "Ollama error: connection refused");
    }

    #[test]
    fn tree_sitter_error_displays() {
        let err = ContextPlusError::TreeSitter("parse failed".into());
        assert_eq!(err.to_string(), "Tree-sitter parse error: parse failed");
    }

    #[test]
    fn path_traversal_error_displays() {
        let err = ContextPlusError::PathTraversal("../etc/passwd".into());
        assert_eq!(err.to_string(), "Path traversal detected: ../etc/passwd");
    }

    #[test]
    fn cache_error_displays() {
        let err = ContextPlusError::Cache("corrupt data".into());
        assert_eq!(err.to_string(), "Cache error: corrupt data");
    }

    #[test]
    fn config_error_displays() {
        let err = ContextPlusError::Config("missing var".into());
        assert_eq!(err.to_string(), "Config error: missing var");
    }

    #[test]
    fn serialization_error_displays() {
        let err = ContextPlusError::Serialization("invalid format".into());
        assert_eq!(err.to_string(), "Serialization error: invalid format");
    }

    #[test]
    fn other_error_displays() {
        let err = ContextPlusError::Other("unknown issue".into());
        assert_eq!(err.to_string(), "unknown issue");
    }

    #[test]
    fn result_type_alias_works() {
        let ok: Result<i32> = Ok(42);
        assert!(ok.is_ok());
        assert_eq!(*ok.as_ref().unwrap(), 42);

        let err: Result<i32> = Err(ContextPlusError::Other("test".into()));
        assert!(err.is_err());
    }
}
