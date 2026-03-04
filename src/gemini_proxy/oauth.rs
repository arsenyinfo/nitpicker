use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use eyre::Result;
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use url::Url;

pub fn generate_pkce_challenge() -> (String, String) {
    let verifier = generate_code_verifier();
    let challenge = generate_code_challenge(&verifier);
    (verifier, challenge)
}

fn generate_code_verifier() -> String {
    let mut random_bytes = [0u8; 64];
    rand::rng().fill_bytes(&mut random_bytes);
    URL_SAFE_NO_PAD.encode(random_bytes)
}

fn generate_code_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let hash = hasher.finalize();
    URL_SAFE_NO_PAD.encode(hash)
}

pub fn build_authorization_url(
    state: &str,
    code_challenge: &str,
    redirect_uri: &str,
) -> Result<String> {
    let mut url = Url::parse(super::OAUTH_AUTH_URL)
        .map_err(|e| eyre::eyre!("Failed to parse OAuth auth URL: {e}"))?;

    let scopes = super::SCOPES.join(" ");

    url.query_pairs_mut()
        .append_pair("client_id", super::CLIENT_ID)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", &scopes)
        .append_pair("state", state)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");

    Ok(url.to_string())
}

#[derive(Debug, serde::Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: i64,
    pub token_type: String,
}

pub async fn exchange_code_for_token(
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<TokenResponse> {
    let client = reqwest::Client::new();

    let mut params = HashMap::new();
    let client_id =
        std::env::var("GEMINI_OAUTH_CLIENT_ID").unwrap_or_else(|_| super::CLIENT_ID.to_string());
    let client_secret = std::env::var("GEMINI_OAUTH_CLIENT_SECRET")
        .unwrap_or_else(|_| super::CLIENT_SECRET.to_string());
    params.insert("client_id", client_id.as_str());
    params.insert("client_secret", client_secret.as_str());
    params.insert("code", code);
    params.insert("code_verifier", code_verifier);
    params.insert("grant_type", "authorization_code");
    params.insert("redirect_uri", redirect_uri);

    let response = client
        .post(super::OAUTH_TOKEN_URL)
        .form(&params)
        .send()
        .await?;

    if !response.status().is_success() {
        let text = response.text().await?;
        eyre::bail!("Token exchange failed: {}", text);
    }

    let token_response = response.json::<TokenResponse>().await?;
    Ok(token_response)
}

pub async fn refresh_access_token(refresh_token: &str) -> Result<TokenResponse> {
    let client = reqwest::Client::new();

    let mut params = HashMap::new();
    let client_id =
        std::env::var("GEMINI_OAUTH_CLIENT_ID").unwrap_or_else(|_| super::CLIENT_ID.to_string());
    let client_secret = std::env::var("GEMINI_OAUTH_CLIENT_SECRET")
        .unwrap_or_else(|_| super::CLIENT_SECRET.to_string());
    params.insert("client_id", client_id.as_str());
    params.insert("client_secret", client_secret.as_str());
    params.insert("refresh_token", refresh_token);
    params.insert("grant_type", "refresh_token");

    let response = client
        .post(super::OAUTH_TOKEN_URL)
        .form(&params)
        .send()
        .await?;

    if !response.status().is_success() {
        let text = response.text().await?;
        eyre::bail!("Token refresh failed: {}", text);
    }

    let token_response = response.json::<TokenResponse>().await?;
    Ok(token_response)
}
