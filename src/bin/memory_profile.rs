use anyhow::{Context, Result};

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize> {
    let Some(position) = args.iter().position(|arg| arg == name) else {
        return Ok(default);
    };
    args.get(position + 1)
        .with_context(|| format!("{name} requires a value"))?
        .parse()
        .with_context(|| format!("invalid value for {name}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(position) = args.iter().position(|arg| arg == "--root") {
        let root = args.get(position + 1).context("--root requires a value")?;
        contextplus_rs::memory_profile::profile_existing_root(root.into())
            .await
            .with_context(|| format!("cannot profile --root {root}"))?;
        return Ok(());
    }
    let linked_refs = parse_usize(&args, "--refs", 5)?;
    let cycles = parse_usize(&args, "--cycles", 20)?;
    contextplus_rs::memory_profile::run_memory_profile(linked_refs, cycles).await;
    Ok(())
}
