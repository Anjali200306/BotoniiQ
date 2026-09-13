import api from "./api";

export interface User {
  id: number;
  name: string;
  email: string;
  role: string;
}

export interface SignupPayload {
  name: string;
  email: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface AuthResponse {
  message: string;
  access_token: string;
  user: User;
}

export const signup = async (
  data: SignupPayload
): Promise<AuthResponse> => {
  const res = await api.post<AuthResponse>("/auth/signup", data);
  return res.data;
};

export const login = async (
  data: LoginPayload
): Promise<AuthResponse> => {
  const res = await api.post<AuthResponse>("/auth/login", data);
  return res.data;
};

export const getCurrentUser = async () => {
  const res = await api.get<{ user: User }>("/auth/me");
  return res.data;
};