//! Token counting utilities for prompt and streaming token counting

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::types::OpenAIChatRequest;
use tiktoken_rs::{CoreBPE, cl100k_base, get_bpe_from_model};

lazy_static::lazy_static! {
    /// Static reference to the default cl100k_base tokenizer (used by all modern OpenAI models)
    static ref DEFAULT_TOKENIZER: CoreBPE = cl100k_base().expect("Failed to initialize cl100k_base tokenizer");
}

/// Count tokens in an OpenAI chat request prompt
///
/// Uses the appropriate tokenizer for the model, or defaults to `cl100k_base`
/// (good approximation for most models) if the model isn't recognized.
pub fn count_prompt_tokens(req: &OpenAIChatRequest) -> usize {
    let tokenizer = get_bpe_from_model(&req.model).unwrap_or_else(|_| DEFAULT_TOKENIZER.clone());

    let mut total_tokens = 0;

    // Count tokens from each message's content
    for message in &req.messages {
        let content = message.content.as_deref().unwrap_or("");
        let tokens = tokenizer.encode_with_special_tokens(content);
        total_tokens += tokens.len();
    }

    total_tokens
}

/// Incremental token counter for streaming responses
///
/// Uses an `AtomicUsize` for thread-safe counting across multiple chunks.
#[derive(Debug, Default)]
pub struct IncrementalTokenCounter {
    count: AtomicUsize,
}

impl IncrementalTokenCounter {
    /// Create a new incremental token counter starting at 0
    pub fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
        }
    }

    /// Add new delta content and return the updated total
    pub fn add_delta(&self, content: &str) -> usize {
        let tokens = DEFAULT_TOKENIZER.encode_with_special_tokens(content);
        let added = tokens.len();
        let prev = self.count.fetch_add(added, Ordering::SeqCst);
        prev + added
    }

    /// Get the current total token count
    pub fn total(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OpenAIMessage;

    fn openai_message(role: &str, content: &str) -> OpenAIMessage {
        OpenAIMessage {
            role: role.to_string(),
            content: Some(content.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn test_count_prompt_tokens_simple() {
        let req = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![openai_message("user", "Hello world")],
            ..Default::default()
        };

        let count = count_prompt_tokens(&req);
        assert!(count > 0);
        assert!(count < 10); // "Hello world" should be ~2 tokens
    }

    #[test]
    fn test_count_prompt_tokens_multiple_messages() {
        let req = OpenAIChatRequest {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![
                openai_message("system", "You are a helpful assistant."),
                openai_message("user", "What is the capital of France?"),
            ],
            ..Default::default()
        };

        let count = count_prompt_tokens(&req);
        assert!(count > 10);
        assert!(count < 30);
        // "You are a helpful assistant." ~ 7 tokens
        // "What is the capital of France?" ~ 8 tokens
        // Total ~15 tokens, so between 10 and 30 is safe
    }

    #[test]
    fn test_count_prompt_tokens_unknown_model() {
        let req = OpenAIChatRequest {
            model: "unknown-model-123".to_string(),
            messages: vec![openai_message("user", "Hello world")],
            ..Default::default()
        };

        let count = count_prompt_tokens(&req);
        assert!(count > 0);
        assert!(count < 10); // Should still work with default tokenizer
    }

    #[test]
    fn test_incremental_token_counter_basic() {
        let counter = IncrementalTokenCounter::new();
        assert_eq!(counter.total(), 0);

        let total = counter.add_delta("Hello");
        assert!(total > 0);
        assert!(total < 3); // "Hello" is 1 token
        assert_eq!(counter.total(), total);

        let total = counter.add_delta(" world");
        assert!(total > 1);
        assert!(total < 4); // "Hello world" is ~2 tokens
        assert_eq!(counter.total(), total);
    }

    #[test]
    fn test_incremental_token_counter_empty() {
        let counter = IncrementalTokenCounter::new();
        let total = counter.add_delta("");
        assert_eq!(total, 0);
        assert_eq!(counter.total(), 0);
    }

    #[test]
    fn test_incremental_token_counter_multiple_deltas() {
        let counter = IncrementalTokenCounter::new();

        counter.add_delta("This ");
        counter.add_delta("is ");
        counter.add_delta("a ");
        counter.add_delta("test.");

        let total = counter.total();
        assert!(total > 0);
        assert!(total < 10); // "This is a test." ~ 5 tokens
    }
}
