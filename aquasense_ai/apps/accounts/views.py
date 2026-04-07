"""
apps/accounts/views.py
Auth API views — register, login, profile.
"""

from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model

from .serializers import (UserRegistrationSerializer,
                           UserProfileSerializer,
                           CustomTokenSerializer)

User = get_user_model()


class RegisterView(APIView):
    """
    POST /api/auth/register/
    Creates a new user account.
    No auth required (AllowAny).
    """
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            # Auto-generate JWT tokens so user is logged in immediately after register
            refresh = RefreshToken.for_user(user)
            return Response({
                'message': 'Account created successfully.',
                'user': UserProfileSerializer(user).data,
                'tokens': {
                    'access':  str(refresh.access_token),
                    'refresh': str(refresh),
                }
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(TokenObtainPairView):
    """
    POST /api/auth/login/
    Returns JWT access + refresh tokens.
    Uses CustomTokenSerializer to include user info.
    """
    permission_classes  = [AllowAny]
    serializer_class    = CustomTokenSerializer


class LogoutView(APIView):
    """
    POST /api/auth/logout/
    Blacklists the refresh token so it can't be reused.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({'message': 'Logged out successfully.'},
                            status=status.HTTP_200_OK)
        except Exception:
            return Response({'error': 'Invalid token.'},
                            status=status.HTTP_400_BAD_REQUEST)


class ProfileView(generics.RetrieveUpdateAPIView):
    """
    GET  /api/auth/profile/  — Get own profile
    PUT  /api/auth/profile/  — Update own profile
    """
    permission_classes = [IsAuthenticated]
    serializer_class   = UserProfileSerializer

    def get_object(self):
        return self.request.user   # Always returns the logged-in user's profile