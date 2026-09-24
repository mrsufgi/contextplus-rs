use super::*;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config(server: &MockServer) -> Config {
    let mut config = Config::from_env();
    config.ollama_host = server.uri();
    config.openai_base_url = format!("{}/v1/", server.uri());
    config.anthropic_base_url = format!("{}/v1", server.uri());
    config.chat_base_url = None;
    config.ollama_api_key = Some("fixture-secret".into());
    config.openai_api_key = Some("fixture-secret".into());
    config.chat_api_key = Some("fixture-secret".into());
    config.anthropic_api_key = Some("fixture-secret".into());
    config.anthropic_auth_token = None;
    config.embed_query_prefix.clear();
    config.embed_doc_prefix.clear();
    config
}

#[tokio::test]
async fn http_errors_never_echo_remote_secrets() {
    for provider in [
        ChatProvider::Ollama,
        ChatProvider::OpenAi,
        ChatProvider::Anthropic,
    ] {
        for malformed in [false, true] {
            let server = MockServer::start().await;
            let response = if malformed {
                match provider {
                    ChatProvider::Ollama => json!({"message":{"content":{"fixture-secret":true}}}),
                    ChatProvider::OpenAi => {
                        json!({"choices":[{"message":{"content":{"fixture-secret":true}}}]})
                    }
                    _ => json!({"content":[],"stop_reason":"fixture-secret"}),
                }
            } else {
                json!({"error":"fixture-secret", "other":"unselected-secret"})
            };
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(if malformed { 200 } else { 401 })
                        .set_body_json(response),
                )
                .expect(1)
                .mount(&server)
                .await;
            let mut cfg = config(&server);
            cfg.chat_provider = provider;
            cfg.anthropic_auth_token = Some("unselected-secret".into());
            let error = OllamaClient::new(&cfg).chat("label").await.unwrap_err();
            let rendered = format!("{error} {error:?}");
            assert!(
                !rendered.contains("fixture-secret"),
                "remote content leaked"
            );
            assert!(
                !rendered.contains("unselected-secret"),
                "unused credential leaked"
            );
        }
    }
    for provider in [EmbedProvider::Ollama, EmbedProvider::OpenAi] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(400).set_body_string("fixture-secret unselected-secret"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let mut cfg = config(&server);
        cfg.embed_provider = provider;
        let error = OllamaClient::new(&cfg).embed_query("q").await.unwrap_err();
        let rendered = format!("{error:?}");
        assert!(
            !rendered.contains("fixture-secret"),
            "remote content leaked"
        );
        assert!(
            !rendered.contains("unselected-secret"),
            "unused credential leaked"
        );
    }
}

#[tokio::test]
async fn anthropic_bearer_error_does_not_echo_token() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bearer-error-fixture"))
        .expect(1)
        .mount(&server)
        .await;
    let mut cfg = config(&server);
    cfg.chat_provider = ChatProvider::Anthropic;
    cfg.anthropic_api_key = None;
    cfg.anthropic_auth_token = Some("bearer-error-fixture".into());
    let error = OllamaClient::new(&cfg).chat("q").await.unwrap_err();
    assert!(!format!("{error:?}").contains("bearer-error-fixture"));
}

#[tokio::test]
async fn transport_errors_do_not_echo_secret_urls() {
    let server = MockServer::start().await;
    let mut cfg = config(&server);
    let url = "http://127.0.0.1:1/fixture-secret";
    cfg.ollama_host = url.into();
    cfg.openai_base_url = url.into();
    cfg.anthropic_base_url = url.into();
    for provider in [
        ChatProvider::Ollama,
        ChatProvider::OpenAi,
        ChatProvider::Anthropic,
    ] {
        cfg.chat_provider = provider;
        let error = OllamaClient::new(&cfg).chat("q").await.unwrap_err();
        assert!(
            !format!("{error:?}").contains("fixture-secret"),
            "transport error leaked URL"
        );
    }
    for provider in [EmbedProvider::Ollama, EmbedProvider::OpenAi] {
        cfg.embed_provider = provider;
        let error = OllamaClient::new(&cfg).embed_query("q").await.unwrap_err();
        assert!(
            !format!("{error:?}").contains("fixture-secret"),
            "transport error leaked URL"
        );
    }
}

#[tokio::test]
async fn invalid_openai_indices_fail_without_caching() {
    for data in [
        json!([]),
        json!([{"index":0,"embedding":[1.0]},{"index":0,"embedding":[2.0]}]),
        json!([{"index":0,"embedding":[1.0]},{"index":2,"embedding":[2.0]}]),
    ] {
        let server = MockServer::start().await;
        Mock::given(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":data})))
            .expect(1)
            .mount(&server)
            .await;
        let mut cfg = config(&server);
        cfg.embed_provider = EmbedProvider::OpenAi;
        let client = OllamaClient::new(&cfg);
        assert!(
            client
                .embed_documents(&["a".into(), "b".into()])
                .await
                .is_err()
        );
        assert_eq!(client.query_cache_len(), 0);
    }
}

#[tokio::test]
async fn long_openai_inputs_merge_and_preserve_prefixes() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/embeddings"))
        .respond_with(|request: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            let data: Vec<_> = body["input"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(index, _)| json!({"index":index,"embedding":[3.0,4.0]}))
                .collect();
            ResponseTemplate::new(200).set_body_json(json!({"data":data}))
        })
        .mount(&server)
        .await;
    let mut cfg = config(&server);
    cfg.embed_provider = EmbedProvider::OpenAi;
    cfg.embed_chunk_chars = 20;
    cfg.embed_batch_size = 2;
    cfg.embed_query_prefix = "query: ".into();
    cfg.embed_doc_prefix = "doc: ".into();
    let client = OllamaClient::new(&cfg);
    let query = client.embed_query(&"x".repeat(50)).await.unwrap();
    assert_eq!(query, vec![3.0, 4.0]);
    assert_eq!(
        client.embed_documents(&["hello".into()]).await.unwrap(),
        vec![vec![3.0, 4.0]]
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        3,
        "three query chunks in two batches and one document"
    );
    let bodies: Vec<serde_json::Value> = requests
        .iter()
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect();
    assert!(
        bodies[0]["input"][0]
            .as_str()
            .unwrap()
            .starts_with("query: ")
    );
    assert_eq!(bodies.last().unwrap()["input"][0], "doc: hello");
    assert!(
        bodies
            .iter()
            .all(|body| body["input"].as_array().unwrap().len() <= 2)
    );
}

#[tokio::test]
async fn single_openai_input_shrinks_after_context_error() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/embeddings"))
        .respond_with(|request: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            if body["input"][0].as_str().unwrap().len() > 10 {
                ResponseTemplate::new(400)
                    .set_body_json(json!({"error":{"code":"context_length_exceeded"}}))
            } else {
                ResponseTemplate::new(200)
                    .set_body_json(json!({"data":[{"index":0,"embedding":[1.0]}]}))
            }
        })
        .mount(&server)
        .await;
    let mut cfg = config(&server);
    cfg.embed_provider = EmbedProvider::OpenAi;
    assert_eq!(
        OllamaClient::new(&cfg)
            .embed_query(&"a".repeat(30))
            .await
            .unwrap(),
        vec![1.0]
    );
    assert!(server.received_requests().await.unwrap().len() > 2);
}

#[tokio::test]
async fn http_timeouts_and_empty_chat_results_return_errors() {
    for provider in [
        ChatProvider::Ollama,
        ChatProvider::OpenAi,
        ChatProvider::Anthropic,
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(2)))
            .mount(&server)
            .await;
        let mut cfg = config(&server);
        cfg.chat_provider = provider;
        let client =
            OllamaClient::new(&cfg).with_chat_timeout(std::time::Duration::from_millis(50));
        assert!(
            client
                .chat("q")
                .await
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );
        server.reset().await;
        let response = match provider {
            ChatProvider::Ollama => json!({"message":{"content":" "}}),
            ChatProvider::OpenAi => json!({"choices":[{"message":{"content":" "}}]}),
            _ => json!({"content":[],"stop_reason":"end_turn"}),
        };
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .mount(&server)
            .await;
        assert!(client.chat("q").await.is_err());
    }
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(2)))
        .mount(&server)
        .await;
    let mut cfg = config(&server);
    cfg.embed_provider = EmbedProvider::OpenAi;
    let client = OllamaClient::new(&cfg).with_request_timeout(std::time::Duration::from_millis(50));
    assert!(
        client
            .embed_query("q")
            .await
            .unwrap_err()
            .to_string()
            .contains("deadline")
    );
}

#[cfg(unix)]
#[tokio::test]
async fn claude_rejects_error_envelopes_and_checks_empty_flag_values() {
    use super::tests::{claude_chat_config, write_fake_claude};
    let dir = tempfile::tempdir().unwrap();
    for envelope in [
        r#"{"is_error":true,"result":"fixture-secret"}"#,
        r#"{"subtype":"error_during_execution","result":"fixture-secret"}"#,
    ] {
        let script = write_fake_claude(dir.path(), &format!("printf '%s' '{envelope}'"));
        let error = OllamaClient::new(&claude_chat_config(&script))
            .chat("q")
            .await
            .unwrap_err();
        assert!(!format!("{error:?}").contains("fixture-secret"));
    }
    let script = write_fake_claude(
        dir.path(),
        r#"
tools=missing
settings=missing
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tools) shift; tools="$1" ;;
    --setting-sources) shift; settings="$1" ;;
  esac
  shift
done
[ "$tools" = "" ] && [ "$settings" = "" ] || exit 8
printf '%s' '{"result":"isolated"}'
"#,
    );
    assert_eq!(
        OllamaClient::new(&claude_chat_config(&script))
            .chat("q")
            .await
            .unwrap(),
        "isolated"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn claude_timeout_kills_child_and_removes_cwd() {
    use super::tests::{claude_chat_config, write_fake_claude};
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("pid");
    let cwd_file = dir.path().join("cwd");
    let script = write_fake_claude(
        dir.path(),
        &format!(
            "echo $$ > '{}'; pwd > '{}'; exec sleep 10",
            pid_file.display(),
            cwd_file.display()
        ),
    );
    let client = OllamaClient::new(&claude_chat_config(&script))
        .with_chat_timeout(std::time::Duration::from_millis(200));
    assert!(client.chat("q").await.is_err());
    let pid: i32 = std::fs::read_to_string(pid_file)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    for _ in 0..100 {
        // SAFETY: signal zero only checks the existence of this test's child.
        if unsafe { libc::kill(pid, 0) } == -1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    // SAFETY: signal zero does not modify the process.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1, "child survived timeout");
    let cwd = std::fs::read_to_string(cwd_file).unwrap();
    assert!(!std::path::Path::new(cwd.trim()).exists());
}

#[tokio::test]
async fn openai_cancellation_and_deadline_cover_semaphore_wait() {
    let server = MockServer::start().await;
    let mut cfg = config(&server);
    cfg.embed_provider = EmbedProvider::OpenAi;
    for cancel in [true, false] {
        let client = OllamaClient::new(&cfg)
            .with_semaphore(Arc::new(tokio::sync::Semaphore::new(0)))
            .with_request_timeout(std::time::Duration::from_millis(50));
        let token = client.cancel_token.clone();
        if cancel {
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                token.cancel();
            });
        }
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            client.embed_query("q"),
        )
        .await
        .expect("semaphore wait ignored cancellation/deadline");
        if cancel {
            assert!(matches!(result, Err(ContextPlusError::Cancelled)));
        } else {
            assert!(result.unwrap_err().to_string().contains("deadline"));
        }
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn openai_persistent_query_cache_isolates_models_and_ollama() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"data":[{"index":0,"embedding":[1.0]}]})),
        )
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings":[[2.0]]})))
        .expect(1)
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(&server);
    cfg.ollama_embed_model = "same".into();
    for (provider, model, vector) in [
        (EmbedProvider::OpenAi, "same", vec![1.0]),
        (EmbedProvider::Ollama, "same", vec![2.0]),
        (EmbedProvider::OpenAi, "another", vec![1.0]),
    ] {
        cfg.embed_provider = provider;
        cfg.openai_embed_model = model.into();
        let client = OllamaClient::new_with_root(&cfg, Some(dir.path().to_path_buf()));
        assert_eq!(client.query_cache_len(), 0);
        assert_eq!(client.embed_query("q").await.unwrap(), vector);
        client.flush_query_cache();
        let restored = OllamaClient::new_with_root(&cfg, Some(dir.path().to_path_buf()));
        assert_eq!(restored.query_cache_len(), 1);
        assert_eq!(restored.embed_query("q").await.unwrap(), vector);
    }
}

#[tokio::test]
async fn separate_chat_endpoint_and_auth_do_not_leak_embedding_key() {
    use wiremock::matchers::header;
    let embed_server = MockServer::start().await;
    let chat_server = MockServer::start().await;
    Mock::given(path("/v1/embeddings"))
        .and(header("authorization", "Bearer fixture-secret"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"data":[{"index":0,"embedding":[1.0]}]})),
        )
        .expect(1)
        .mount(&embed_server)
        .await;
    Mock::given(path("/chat/completions"))
        .and(header("authorization", "Bearer separate-chat-fixture"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"choices":[{"message":{"content":"label"}}]})),
        )
        .expect(1)
        .mount(&chat_server)
        .await;
    let mut cfg = config(&embed_server);
    cfg.embed_provider = EmbedProvider::OpenAi;
    cfg.chat_provider = ChatProvider::OpenAi;
    cfg.chat_base_url = Some(format!("{}/", chat_server.uri()));
    cfg.chat_api_key = Some("separate-chat-fixture".into());
    let client = OllamaClient::new(&cfg);
    assert_eq!(client.embed_query("q").await.unwrap(), vec![1.0]);
    assert_eq!(client.chat("q").await.unwrap(), "label");
}

#[cfg(unix)]
#[tokio::test]
async fn claude_inherits_parent_environment_without_loading_credentials() {
    const CHILD: &str = "CONTEXTPLUS_TEST_CLAUDE_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let cfg = Config::from_env();
        assert_eq!(cfg.chat_provider, ChatProvider::Claude);
        assert!(cfg.anthropic_api_key.is_none());
        assert!(cfg.anthropic_auth_token.is_none());
        assert_eq!(
            OllamaClient::new(&cfg).chat("q").await.unwrap(),
            "inherited"
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let script = super::tests::write_fake_claude(
        dir.path(),
        r#"
[ "$CLAUDE_CODE_OAUTH_TOKEN" = "oauth-fixture" ] || exit 10
[ "$ANTHROPIC_API_KEY" = "api-fixture" ] || exit 11
[ "$ANTHROPIC_AUTH_TOKEN" = "bearer-fixture" ] || exit 12
[ "$HOME" = "$CONTEXTPLUS_TEST_EXPECTED_HOME" ] || exit 13
[ "$HTTPS_PROXY" = "http://proxy.example:8080" ] || exit 14
printf '%s' '{"type":"result","subtype":"success","is_error":false,"result":"inherited"}'
"#,
    );
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "core::embeddings::provider_tests::claude_inherits_parent_environment_without_loading_credentials"])
        .env(CHILD, "1")
        .env("CONTEXTPLUS_CHAT_PROVIDER", "claude")
        .env("CONTEXTPLUS_EMBED_PROVIDER", "ollama")
        .env("CONTEXTPLUS_CLAUDE_PATH", script)
        .env("CLAUDE_CODE_OAUTH_TOKEN", "oauth-fixture")
        .env("ANTHROPIC_API_KEY", "api-fixture")
        .env("ANTHROPIC_AUTH_TOKEN", "bearer-fixture")
        .env("HOME", dir.path())
        .env("CONTEXTPLUS_TEST_EXPECTED_HOME", dir.path())
        .env("HTTPS_PROXY", "http://proxy.example:8080")
        .output().unwrap();
    assert!(
        result.status.success(),
        "isolated environment inheritance test failed"
    );
}
