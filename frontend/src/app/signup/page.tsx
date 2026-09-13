"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import AuthLayout from "../../components/AuthLayout";
import { signup } from "../../services/auth";

export default function SignupPage() {
  const router = useRouter();
  const [form, setForm] = useState({ name: "", email: "", password: "" });
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
      const data = await signup(form);
      localStorage.setItem("access_token", data.access_token);
      router.push("/dashboard");
    } catch (err: any) {
      setError(err.response?.data?.error || "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Create your account"
      subtitle="Start identifying and caring for your plants."
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-[family-name:var(--font-inter)] text-[#14201A] mb-1">
            Name
          </label>
          <input
            type="text"
            name="name"
            required
            value={form.name}
            onChange={handleChange}
            className="w-full px-4 py-2.5 rounded-lg border border-[#D9E0D4] bg-white focus:outline-none focus:ring-2 focus:ring-[#C98A3B] font-[family-name:var(--font-inter)] text-sm"
            placeholder="Ashlesha Mandhare"
          />
        </div>

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
            minLength={6}
            value={form.password}
            onChange={handleChange}
            className="w-full px-4 py-2.5 rounded-lg border border-[#D9E0D4] bg-white focus:outline-none focus:ring-2 focus:ring-[#C98A3B] font-[family-name:var(--font-inter)] text-sm"
            placeholder="At least 6 characters"
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
          {loading ? "Creating account..." : "Create account"}
        </button>
      </form>

      <p className="mt-6 text-sm text-[#5A6B58] font-[family-name:var(--font-inter)]">
        Already have an account?{" "}
        <Link href="/login" className="text-[#C98A3B] font-medium hover:underline">
          Log in
        </Link>
      </p>
    </AuthLayout>
  );
}