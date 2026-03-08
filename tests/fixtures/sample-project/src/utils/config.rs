// Configuration loader for the application
// Reads from environment variables with fallbacks

use std::env;

pub struct Config {
    pub database_url: String,
    pub port: u16,
    pub log_level: String,
    pub max_connections: u32,
}

pub fn load_config() -> Config {
    Config {
        database_url: env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost:5432/app".to_string()),
        port: env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000),
        log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        max_connections: env::var("MAX_CONNECTIONS")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(10),
    }
}

pub fn validate_config(config: &Config) -> Result<(), String> {
    if config.database_url.is_empty() {
        return Err("DATABASE_URL cannot be empty".to_string());
    }
    if config.port == 0 {
        return Err("PORT must be non-zero".to_string());
    }
    Ok(())
}
