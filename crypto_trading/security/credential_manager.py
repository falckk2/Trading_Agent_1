"""
Secure credential management with encryption and secure storage.
Provides safe handling of API keys, secrets, and sensitive configuration.
"""

import os
import json
import base64
import hashlib
from typing import Dict, Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger

from ..core.exceptions import SecurityError, ConfigurationError


class SecureCredentialManager:
    """Secure credential management with encryption."""

    def __init__(self, credentials_file: str = "config/credentials.enc", master_password: str = None):
        self.credentials_file = Path(credentials_file)
        self.credentials_file.parent.mkdir(parents=True, exist_ok=True)

        self._master_password = master_password
        self._encryption_key: Optional[bytes] = None
        self._credentials: Dict[str, Any] = {}
        self._is_unlocked = False

        # Salt for key derivation
        self.salt_file = self.credentials_file.with_suffix('.salt')

    def set_master_password(self, password: str) -> None:
        """Set the master password for encryption."""
        if not password or len(password) < 8:
            raise SecurityError("Master password must be at least 8 characters long")

        self._master_password = password
        self._encryption_key = self._derive_key(password)

    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        # Load or create salt
        if self.salt_file.exists():
            with open(self.salt_file, 'rb') as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(self.salt_file, 'wb') as f:
                f.write(salt)

        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def unlock(self, password: str = None) -> bool:
        """Unlock credentials with master password."""
        if password:
            self.set_master_password(password)

        if not self._master_password:
            raise SecurityError("Master password not set")

        try:
            self._encryption_key = self._derive_key(self._master_password)

            # Try to load and decrypt credentials to verify password
            if self.credentials_file.exists():
                self._load_encrypted_credentials()

            self._is_unlocked = True
            logger.info("Credential manager unlocked successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unlock credentials: {e}")
            self._is_unlocked = False
            return False

    def lock(self) -> None:
        """Lock credentials and clear from memory."""
        self._credentials.clear()
        self._encryption_key = None
        self._master_password = None
        self._is_unlocked = False
        logger.info("Credential manager locked")

    def store_credential(self, key: str, value: Any) -> None:
        """Store a credential securely."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        self._credentials[key] = value
        self._save_encrypted_credentials()
        logger.info(f"Stored credential: {key}")

    def get_credential(self, key: str, default: Any = None) -> Any:
        """Retrieve a credential."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        return self._credentials.get(key, default)

    def delete_credential(self, key: str) -> bool:
        """Delete a credential."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        if key in self._credentials:
            del self._credentials[key]
            self._save_encrypted_credentials()
            logger.info(f"Deleted credential: {key}")
            return True

        return False

    def list_credentials(self) -> list[str]:
        """List all credential keys (not values)."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        return list(self._credentials.keys())

    def update_credentials(self, credentials: Dict[str, Any]) -> None:
        """Update multiple credentials at once."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        self._credentials.update(credentials)
        self._save_encrypted_credentials()
        logger.info(f"Updated {len(credentials)} credentials")

    def _save_encrypted_credentials(self) -> None:
        """Save credentials in encrypted format."""
        if not self._encryption_key:
            raise SecurityError("No encryption key available")

        try:
            # Serialize credentials
            data = json.dumps(self._credentials, indent=2)

            # Encrypt data
            fernet = Fernet(self._encryption_key)
            encrypted_data = fernet.encrypt(data.encode())

            # Save to file
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            os.chmod(self.credentials_file, 0o600)

        except Exception as e:
            logger.error(f"Failed to save encrypted credentials: {e}")
            raise SecurityError(f"Failed to save credentials: {e}")

    def _load_encrypted_credentials(self) -> None:
        """Load credentials from encrypted file."""
        if not self._encryption_key:
            raise SecurityError("No encryption key available")

        if not self.credentials_file.exists():
            self._credentials = {}
            return

        try:
            # Read encrypted data
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()

            # Decrypt data
            fernet = Fernet(self._encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)

            # Parse JSON
            self._credentials = json.loads(decrypted_data.decode())

        except Exception as e:
            logger.error(f"Failed to load encrypted credentials: {e}")
            raise SecurityError(f"Failed to load credentials (invalid password?): {e}")

    def export_credentials(self, export_password: str, export_file: str) -> None:
        """Export credentials to another encrypted file."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        try:
            # Create temporary manager with export password
            export_manager = SecureCredentialManager(export_file)
            export_manager.set_master_password(export_password)
            export_manager._is_unlocked = True
            export_manager._credentials = self._credentials.copy()
            export_manager._save_encrypted_credentials()

            logger.info(f"Credentials exported to {export_file}")

        except Exception as e:
            logger.error(f"Failed to export credentials: {e}")
            raise SecurityError(f"Export failed: {e}")

    def import_credentials(self, import_password: str, import_file: str, merge: bool = True) -> None:
        """Import credentials from another encrypted file."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        try:
            # Create temporary manager to read import file
            import_manager = SecureCredentialManager(import_file)
            import_manager.unlock(import_password)

            if merge:
                # Merge with existing credentials
                self._credentials.update(import_manager._credentials)
            else:
                # Replace all credentials
                self._credentials = import_manager._credentials.copy()

            self._save_encrypted_credentials()
            logger.info(f"Credentials imported from {import_file}")

        except Exception as e:
            logger.error(f"Failed to import credentials: {e}")
            raise SecurityError(f"Import failed: {e}")

    def change_master_password(self, old_password: str, new_password: str) -> None:
        """Change the master password."""
        if not self._is_unlocked:
            # Try to unlock with old password
            if not self.unlock(old_password):
                raise SecurityError("Invalid old password")

        # Validate new password
        if not new_password or len(new_password) < 8:
            raise SecurityError("New password must be at least 8 characters long")

        try:
            # Set new password and re-encrypt
            self.set_master_password(new_password)
            self._save_encrypted_credentials()

            logger.info("Master password changed successfully")

        except Exception as e:
            logger.error(f"Failed to change master password: {e}")
            raise SecurityError(f"Failed to change password: {e}")

    def get_exchange_credentials(self, exchange_name: str) -> Dict[str, str]:
        """Get exchange-specific credentials."""
        exchange_key = f"exchange.{exchange_name}"
        credentials = self.get_credential(exchange_key, {})

        # Validate required fields
        required_fields = ['api_key', 'api_secret']
        for field in required_fields:
            if field not in credentials:
                raise ConfigurationError(f"Missing {field} for exchange {exchange_name}")

        return credentials

    def store_exchange_credentials(
        self,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        passphrase: str = None,
        additional_params: Dict[str, str] = None
    ) -> None:
        """Store exchange-specific credentials."""
        credentials = {
            'api_key': api_key,
            'api_secret': api_secret
        }

        if passphrase:
            credentials['passphrase'] = passphrase

        if additional_params:
            credentials.update(additional_params)

        exchange_key = f"exchange.{exchange_name}"
        self.store_credential(exchange_key, credentials)

    def verify_integrity(self) -> bool:
        """Verify the integrity of stored credentials."""
        if not self._is_unlocked:
            raise SecurityError("Credential manager is locked")

        try:
            # Try to decrypt and parse the credentials file
            if self.credentials_file.exists():
                self._load_encrypted_credentials()
                return True
            else:
                return True  # No file means no corruption

        except Exception as e:
            logger.error(f"Credential integrity check failed: {e}")
            return False

    def create_backup(self, backup_file: str = None) -> str:
        """Create a backup of encrypted credentials."""
        if not backup_file:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"config/credentials_backup_{timestamp}.enc"

        backup_path = Path(backup_file)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.credentials_file.exists():
                import shutil
                shutil.copy2(self.credentials_file, backup_path)
                shutil.copy2(self.salt_file, backup_path.with_suffix('.salt'))

                logger.info(f"Credential backup created: {backup_path}")
                return str(backup_path)
            else:
                raise SecurityError("No credentials file to backup")

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise SecurityError(f"Backup failed: {e}")

    def get_security_status(self) -> Dict[str, Any]:
        """Get security status information."""
        return {
            'is_unlocked': self._is_unlocked,
            'credentials_file_exists': self.credentials_file.exists(),
            'salt_file_exists': self.salt_file.exists(),
            'credential_count': len(self._credentials) if self._is_unlocked else 0,
            'file_permissions': oct(os.stat(self.credentials_file).st_mode)[-3:] if self.credentials_file.exists() else None
        }


class EnvironmentCredentialProvider:
    """Credential provider that uses environment variables."""

    def __init__(self, prefix: str = "CRYPTO_TRADING_"):
        self.prefix = prefix

    def get_credential(self, key: str, default: str = None) -> Optional[str]:
        """Get credential from environment variable."""
        env_key = f"{self.prefix}{key.upper()}"
        return os.getenv(env_key, default)

    def get_exchange_credentials(self, exchange_name: str) -> Dict[str, str]:
        """Get exchange credentials from environment variables."""
        exchange_prefix = f"{self.prefix}{exchange_name.upper()}_"

        credentials = {}
        for key in ['API_KEY', 'API_SECRET', 'PASSPHRASE']:
            env_key = f"{exchange_prefix}{key}"
            value = os.getenv(env_key)
            if value:
                credentials[key.lower()] = value

        if not credentials.get('api_key') or not credentials.get('api_secret'):
            raise ConfigurationError(f"Missing required environment variables for {exchange_name}")

        return credentials


class HybridCredentialManager:
    """Hybrid credential manager that tries multiple sources."""

    def __init__(
        self,
        secure_manager: SecureCredentialManager = None,
        env_provider: EnvironmentCredentialProvider = None,
        prefer_env: bool = False
    ):
        self.secure_manager = secure_manager or SecureCredentialManager()
        self.env_provider = env_provider or EnvironmentCredentialProvider()
        self.prefer_env = prefer_env

    def get_credential(self, key: str, default: Any = None) -> Any:
        """Get credential from the best available source."""
        sources = [self.env_provider, self.secure_manager] if self.prefer_env else [self.secure_manager, self.env_provider]

        for source in sources:
            try:
                value = source.get_credential(key, None)
                if value is not None:
                    return value
            except Exception as e:
                logger.debug(f"Failed to get credential {key} from {type(source).__name__}: {e}")

        return default

    def get_exchange_credentials(self, exchange_name: str) -> Dict[str, str]:
        """Get exchange credentials from the best available source."""
        sources = [self.env_provider, self.secure_manager] if self.prefer_env else [self.secure_manager, self.env_provider]

        for source in sources:
            try:
                credentials = source.get_exchange_credentials(exchange_name)
                if credentials:
                    logger.info(f"Using credentials from {type(source).__name__} for {exchange_name}")
                    return credentials
            except Exception as e:
                logger.debug(f"Failed to get exchange credentials from {type(source).__name__}: {e}")

        raise ConfigurationError(f"No valid credentials found for exchange {exchange_name}")

    def unlock_secure_manager(self, password: str) -> bool:
        """Unlock the secure credential manager."""
        return self.secure_manager.unlock(password)

    def is_secure_manager_available(self) -> bool:
        """Check if secure manager is unlocked and available."""
        return self.secure_manager._is_unlocked


# Factory functions for easy setup
def create_secure_credential_manager(credentials_file: str = "config/credentials.enc") -> SecureCredentialManager:
    """Create a secure credential manager."""
    return SecureCredentialManager(credentials_file)


def create_hybrid_credential_manager(
    credentials_file: str = "config/credentials.enc",
    prefer_env: bool = False
) -> HybridCredentialManager:
    """Create a hybrid credential manager."""
    secure_manager = SecureCredentialManager(credentials_file)
    env_provider = EnvironmentCredentialProvider()
    return HybridCredentialManager(secure_manager, env_provider, prefer_env)