"""
apps/accounts/serializers.py
Serializers convert Python objects ↔ JSON for the API.
"""

from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth import get_user_model

User = get_user_model()


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Handles new user signup. Validates password and creates user."""

    password  = serializers.CharField(write_only=True, min_length=8)
    password2 = serializers.CharField(write_only=True, label="Confirm password")

    class Meta:
        model  = User
        fields = ['email', 'username', 'password', 'password2',
                  'first_name', 'last_name', 'phone', 'city', 'state']

    def validate(self, data):
        if data['password'] != data['password2']:
            raise serializers.ValidationError(
                {"password2": "Passwords do not match."})
        return data

    def create(self, validated_data):
        validated_data.pop('password2')
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)   # Hashes the password
        user.save()
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    """Returns public user profile data."""

    class Meta:
        model  = User
        fields = ['id', 'email', 'username', 'first_name', 'last_name',
                  'phone', 'city', 'state', 'latitude', 'longitude',
                  'total_reports', 'verified_citizen', 'date_joined']
        read_only_fields = ['id', 'email', 'total_reports',
                            'verified_citizen', 'date_joined']


class CustomTokenSerializer(TokenObtainPairSerializer):
    """
    Extends the JWT token response to include user info.
    Default JWT only returns access + refresh tokens.
    We add: user_id, email, city so the frontend can personalise immediately.
    """

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        # Add custom claims to the JWT payload
        token['email'] = user.email
        token['city']  = user.city
        token['username'] = user.username
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        # Add extra fields to the login response body
        data['user'] = {
            'id':       self.user.id,
            'email':    self.user.email,
            'username': self.user.username,
            'city':     self.user.city,
        }
        return data