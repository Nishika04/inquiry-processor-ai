"""
tests/test_models.py

Unit tests for Pydantic model validation in models.py.
No LLM calls — purely validates schema constraints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pydantic import ValidationError
from models import InquiryFormInput


class TestInquiryFormInput:
    def _valid_payload(self, **overrides):
        base = {
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice@example.com",
            "subject": "Hello",
            "message": "I would like to know more.",
        }
        base.update(overrides)
        return base

    def test_valid_full(self):
        form = InquiryFormInput(**self._valid_payload())
        assert form.email == "alice@example.com"

    def test_email_lowercased(self):
        form = InquiryFormInput(**self._valid_payload(email="ALICE@EXAMPLE.COM"))
        assert form.email == "alice@example.com"

    def test_invalid_email_no_at(self):
        with pytest.raises(ValidationError):
            InquiryFormInput(**self._valid_payload(email="notanemail"))

    def test_invalid_email_no_tld(self):
        with pytest.raises(ValidationError):
            InquiryFormInput(**self._valid_payload(email="alice@example"))

    def test_empty_first_name(self):
        with pytest.raises(ValidationError):
            InquiryFormInput(**self._valid_payload(first_name="   "))

    def test_empty_message(self):
        with pytest.raises(ValidationError):
            InquiryFormInput(**self._valid_payload(message=""))

    def test_optional_phone_none(self):
        form = InquiryFormInput(**self._valid_payload())
        assert form.phone is None

    def test_optional_phone_provided(self):
        form = InquiryFormInput(**self._valid_payload(phone="+1-800-555-1234"))
        assert form.phone == "+1-800-555-1234"

    def test_default_contact_method(self):
        form = InquiryFormInput(**self._valid_payload())
        assert form.preferred_contact_method == "email"
