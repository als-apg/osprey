"""Tests for AI model factory."""

from osprey.models.factory import _validate_proxy_url

# =============================================================================
# Test Proxy URL Validation
# =============================================================================


class TestProxyUrlValidation:
    """Test HTTP proxy URL validation."""

    def test_validate_valid_http_proxy(self):
        """Test validation of valid HTTP proxy URL."""
        assert _validate_proxy_url("http://proxy.example.com:8080") is True

    def test_validate_valid_https_proxy(self):
        """Test validation of valid HTTPS proxy URL."""
        assert _validate_proxy_url("https://proxy.example.com:8443") is True

    def test_validate_proxy_with_auth(self):
        """Test validation of proxy URL with authentication."""
        assert _validate_proxy_url("http://user:pass@proxy.example.com:8080") is True

    def test_validate_proxy_with_ipv4(self):
        """Test validation of proxy URL with IPv4 address."""
        assert _validate_proxy_url("http://192.168.1.1:8080") is True

    def test_validate_proxy_https_default_port(self):
        """Test validation of HTTPS proxy without explicit port."""
        assert _validate_proxy_url("https://proxy.example.com") is True

    def test_validate_empty_proxy_url(self):
        """Test validation rejects empty string."""
        assert _validate_proxy_url("") is False

    def test_validate_none_proxy_url(self):
        """Test validation rejects None."""
        assert _validate_proxy_url(None) is False

    def test_validate_invalid_scheme(self):
        """Test validation rejects invalid scheme."""
        assert _validate_proxy_url("ftp://proxy.example.com:8080") is False

    def test_validate_socks_scheme(self):
        """Test validation rejects SOCKS proxy scheme."""
        assert _validate_proxy_url("socks5://proxy.example.com:1080") is False

    def test_validate_no_scheme(self):
        """Test validation rejects URL without scheme."""
        assert _validate_proxy_url("proxy.example.com:8080") is False

    def test_validate_no_netloc(self):
        """Test validation rejects URL without netloc."""
        assert _validate_proxy_url("http://") is False

    def test_validate_malformed_url(self):
        """Test validation handles malformed URLs gracefully."""
        assert _validate_proxy_url("not a valid url at all") is False

    def test_validate_proxy_with_path(self):
        """Test validation accepts proxy URL with path component."""
        assert _validate_proxy_url("http://proxy.example.com:8080/path") is True
