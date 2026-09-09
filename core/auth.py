"""
Registration and login.

Passwords are never stored in plain text - each user gets a random salt and the
password is stretched with PBKDF2-HMAC-SHA256 (200k iterations), which is the
same family of algorithm Django uses by default.
"""

import hashlib
import hmac
import os
import re

from core import database as db

_ITERATIONS = 200_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS
    )
    return digest.hex()


def register(name: str, email: str, password: str) -> tuple[bool, str]:
    """Create a new account. Returns (success, message)."""
    name, email = name.strip(), email.strip().lower()

    if not name or not email or not password:
        return False, "All fields are required."
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if db.query_one("SELECT id FROM users WHERE email = ?", (email,)):
        return False, "An account with that email already exists."

    salt = os.urandom(16).hex()
    db.execute(
        """INSERT INTO users (name, email, password_hash, salt, created_at)
           VALUES (?,?,?,?,?)""",
        (name, email, _hash_password(password, salt), salt, db.now()),
    )
    return True, "Account created. You can now sign in."


def login(email: str, password: str) -> tuple[bool, str, dict | None]:
    """Verify credentials. Returns (success, message, user)."""
    email = email.strip().lower()
    user = db.query_one("SELECT * FROM users WHERE email = ?", (email,))

    if not user:
        return False, "No account found for that email.", None

    # hmac.compare_digest avoids leaking information through timing.
    expected = _hash_password(password, user["salt"])
    if not hmac.compare_digest(expected, user["password_hash"]):
        return False, "Incorrect password.", None

    return True, f"Welcome back, {user['name']}.", {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
    }
