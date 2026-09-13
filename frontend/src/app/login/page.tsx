"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import AuthLayout from "../../components/AuthLayout";
import { login } from "../../services/auth";

export default function LoginPage() {
  const router = useRouter();
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  setError("");
  setLoading(true);

  try {
    const data = await login(form);

    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("user", JSON.stringify(data.user));

    router.push("/dashboard");
  } catch (err: any) {
    setError(
      err.response?.data?.error ||
        "Unable to login. Please check your email and password."
    );
  } finally {
    setLoading(false);
  }
};

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Log in to continue caring for your plants."
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-[family-name:var(--font-inter)] text-[#14201A] mb-1">
            Email
          </label>
          <input
            type="email"
            name="email"
            required
            value={form.email}
            onChange={handleChange}
            className="w-full px-4 py-2.5 rounded-lg border border-[#D9E0D4] bg-white focus:outline-none focus:ring-2 focus:ring-[#C98A3B] font-[family-name:var(--font-inter)] text-sm"
            placeholder="you@example.com"
          />
        </div>

        <div>
          <label className="block text-sm font-[family-name:var(--font-inter)] text-[#14201A] mb-1">
            Password
          </label>
          <input
            type="password"
            name="password"
            required
            value={form.password}
            onChange={handleChange}
            className="w-full px-4 py-2.5 rounded-lg border border-[#D9E0D4] bg-white focus:outline-none focus:ring-2 focus:ring-[#C98A3B] font-[family-name:var(--font-inter)] text-sm"
            placeholder="Your password"
          />
        </div>

        {error && (
          <p className="text-sm text-red-600 font-[family-name:var(--font-inter)]">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-[#1E3A2E] text-white py-2.5 rounded-lg font-[family-name:var(--font-inter)] text-sm font-medium hover:bg-[#14201A] transition-colors disabled:opacity-60"
        >
          {loading ? "Logging in..." : "Log in"}
        </button>
      </form>

      <p className="mt-6 text-sm text-[#5A6B58] font-[family-name:var(--font-inter)]">
        Don&apos;t have an account?{" "}
        <Link href="/signup" className="text-[#C98A3B] font-medium hover:underline">
          Sign up
        </Link>
      </p>
    </AuthLayout>
  );
}