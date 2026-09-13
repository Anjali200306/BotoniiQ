"use client";

import {
  ChangeEvent,
  useEffect,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import {
  getCurrentUser,
  User,
} from "../../services/auth";

import {
  scanPlant,
  ScanResult,
} from "../../services/scan";

export default function DashboardPage() {
  const router = useRouter();

  const [user, setUser] = useState<User | null>(null);

  const [selectedImage, setSelectedImage] =
    useState<File | null>(null);

  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  const [scanResult, setScanResult] =
    useState<ScanResult | null>(null);

  const [loadingUser, setLoadingUser] =
    useState(true);

  const [scanning, setScanning] =
    useState(false);

  const [error, setError] =
    useState("");

  // ============================================================
  // LOAD CURRENT USER
  // ============================================================

  useEffect(() => {
    const loadUser = async () => {
      const token = localStorage.getItem(
        "access_token"
      );

      if (!token) {
        router.replace("/login");
        return;
      }

      try {
        const data = await getCurrentUser();

        setUser(data.user);

        localStorage.setItem(
          "user",
          JSON.stringify(data.user)
        );
      } catch {
        localStorage.removeItem(
          "access_token"
        );

        localStorage.removeItem("user");

        router.replace("/login");
      } finally {
        setLoadingUser(false);
      }
    };

    loadUser();
  }, [router]);

  // ============================================================
  // SELECT IMAGE
  // ============================================================

  const handleImageChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    setError("");
    setScanResult(null);

    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    if (!allowedTypes.includes(file.type)) {
      setError(
        "Please select a JPG, JPEG, PNG, or WEBP image."
      );

      event.target.value = "";
      return;
    }

    const maxSize =
      10 * 1024 * 1024;

    if (file.size > maxSize) {
      setError(
        "Image size must be less than 10 MB."
      );

      event.target.value = "";
      return;
    }

    setSelectedImage(file);

    const url = URL.createObjectURL(file);

    setPreviewUrl(url);
  };

  // ============================================================
  // SCAN PLANT
  // ============================================================

  const handleScan = async () => {
    if (!selectedImage) {
      setError(
        "Please select a plant image first."
      );

      return;
    }

    setError("");
    setScanning(true);
    setScanResult(null);

    try {
      const response = await scanPlant(
        selectedImage
      );

      setScanResult(response.scan);
    } catch (err: any) {
      console.error("Plant scan error:", err);

      const backendMessage =
        err.response?.data?.error ||
        err.response?.data?.message;

      setError(
        backendMessage ||
          "Unable to scan the plant. Please try again."
      );
    } finally {
      setScanning(false);
    }
  };

  // ============================================================
  // CLEAR SCAN
  // ============================================================

  const handleClear = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setScanResult(null);
    setError("");

    const input =
      document.getElementById(
        "plant-image"
      ) as HTMLInputElement | null;

    if (input) {
      input.value = "";
    }
  };

  // ============================================================
  // LOGOUT
  // ============================================================

  const handleLogout = () => {
    localStorage.removeItem(
      "access_token"
    );

    localStorage.removeItem("user");

    router.replace("/login");
  };

  // ============================================================
  // LOADING USER
  // ============================================================

  if (loadingUser) {
    return (
      <main className="min-h-screen flex items-center justify-center bg-[#F7F5EF]">
        <p className="text-[#5A6B58]">
          Loading your dashboard...
        </p>
      </main>
    );
  }

  if (!user) {
    return null;
  }

  // ============================================================
  // DASHBOARD
  // ============================================================

  return (
    <main className="min-h-screen bg-[#F7F5EF]">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="border-b border-[#D9E0D4] bg-white">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">

          <div>
            <h1 className="text-2xl font-semibold text-[#1E3A2E]">
              BotoniiQ
            </h1>

            <p className="text-sm text-[#5A6B58]">
              Intelligent plant health assistant
            </p>
          </div>

          <button
            onClick={handleLogout}
            className="px-4 py-2 rounded-lg border border-[#D9E0D4] text-[#1E3A2E] hover:bg-[#F3F5EF] transition-colors"
          >
            Logout
          </button>

        </div>
      </header>

      {/* ======================================================
          MAIN
      ====================================================== */}

      <section className="max-w-6xl mx-auto px-6 py-10">

        {/* Welcome */}

        <div className="mb-8">

          <p className="text-sm text-[#5A6B58]">
            Welcome back
          </p>

          <h2 className="mt-1 text-3xl font-semibold text-[#14201A]">
            {user.name}
          </h2>

          <p className="mt-2 text-[#5A6B58]">
            Upload a plant image to identify its
            species and check disease information.
          </p>

        </div>

        {/* ====================================================
            SCAN CARD
        ==================================================== */}

        <div className="bg-white rounded-2xl border border-[#D9E0D4] p-8">

          <div className="mb-6">

            <h3 className="text-xl font-semibold text-[#14201A]">
              Scan your plant
            </h3>

            <p className="mt-1 text-sm text-[#5A6B58]">
              Upload a clear image of the plant leaf.
            </p>

          </div>

          {/* ==================================================
              IMAGE UPLOAD AREA
          ================================================== */}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

            {/* Upload */}

            <div>

              <label
                htmlFor="plant-image"
                className="block cursor-pointer"
              >

                <div className="min-h-[280px] rounded-xl border-2 border-dashed border-[#D9E0D4] hover:border-[#C98A3B] bg-[#FAFAF7] flex flex-col items-center justify-center p-6 transition-colors">

                  {!previewUrl ? (
                    <>
                      <div className="w-16 h-16 rounded-full bg-[#EAF0E6] flex items-center justify-center text-3xl">
                        🌿
                      </div>

                      <p className="mt-4 text-[#1E3A2E] font-medium">
                        Choose plant image
                      </p>

                      <p className="mt-2 text-sm text-[#5A6B58] text-center">
                        JPG, PNG or WEBP
                        <br />
                        Maximum size: 10 MB
                      </p>
                    </>
                  ) : (
                    <img
                      src={previewUrl}
                      alt="Selected plant"
                      className="max-h-[250px] max-w-full object-contain rounded-lg"
                    />
                  )}

                </div>

              </label>

              <input
                id="plant-image"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleImageChange}
                className="hidden"
              />

              {selectedImage && (
                <p className="mt-3 text-sm text-[#5A6B58] break-all">
                  Selected:{" "}
                  {selectedImage.name}
                </p>
              )}

              {/* Buttons */}

              <div className="mt-5 flex gap-3">

                <button
                  type="button"
                  onClick={handleScan}
                  disabled={
                    !selectedImage ||
                    scanning
                  }
                  className="flex-1 bg-[#1E3A2E] text-white py-3 rounded-lg font-medium hover:bg-[#14201A] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {scanning
                    ? "Analyzing plant..."
                    : "Scan Plant"}
                </button>

                {(selectedImage ||
                  scanResult) && (
                  <button
                    type="button"
                    onClick={handleClear}
                    disabled={scanning}
                    className="px-5 py-3 rounded-lg border border-[#D9E0D4] text-[#1E3A2E] hover:bg-[#F3F5EF] transition-colors disabled:opacity-50"
                  >
                    Clear
                  </button>
                )}

              </div>

            </div>

            {/* ==================================================
                RESULT
            ================================================== */}

            <div>

              {!scanResult && !scanning && (
                <div className="min-h-[280px] rounded-xl border border-[#D9E0D4] bg-[#FAFAF7] flex flex-col items-center justify-center text-center p-6">

                  <div className="text-4xl">
                    🔍
                  </div>

                  <p className="mt-4 font-medium text-[#1E3A2E]">
                    Scan result
                  </p>

                  <p className="mt-2 text-sm text-[#5A6B58] max-w-sm">
                    Your plant identification
                    and disease analysis will
                    appear here.
                  </p>

                </div>
              )}

              {scanning && (
                <div className="min-h-[280px] rounded-xl border border-[#D9E0D4] bg-[#FAFAF7] flex flex-col items-center justify-center text-center p-6">

                  <div className="text-4xl animate-pulse">
                    🌱
                  </div>

                  <p className="mt-4 font-medium text-[#1E3A2E]">
                    Analyzing your plant...
                  </p>

                  <p className="mt-2 text-sm text-[#5A6B58]">
                    Running species and disease
                    analysis.
                  </p>

                </div>
              )}

              {scanResult && (
                <div className="space-y-4">

                  {/* Species */}

                  <div className="rounded-xl border border-[#D9E0D4] p-5">

                    <p className="text-sm text-[#5A6B58]">
                      Identified species
                    </p>

                    <div className="mt-1 flex items-center justify-between gap-4">

                      <h4 className="text-2xl font-semibold text-[#1E3A2E]">
                        {scanResult.species.name}
                      </h4>

                      <span className="text-sm font-medium text-[#C98A3B]">
                        {scanResult.species.confidence.toFixed(
                          2
                        )}
                        %
                      </span>

                    </div>

                  </div>

                  {/* Top 3 */}

                  <div className="rounded-xl border border-[#D9E0D4] p-5">

                    <p className="text-sm font-medium text-[#14201A]">
                      Top predictions
                    </p>

                    <div className="mt-3 space-y-3">

                      {scanResult.species.top_3.map(
                        (prediction, index) => (
                          <div
                            key={`${prediction.name}-${index}`}
                            className="flex items-center justify-between text-sm"
                          >

                            <span className="text-[#5A6B58]">
                              {index + 1}.{" "}
                              {prediction.name}
                            </span>

                            <span className="font-medium text-[#1E3A2E]">
                              {prediction.confidence.toFixed(
                                2
                              )}
                              %
                            </span>

                          </div>
                        )
                      )}

                    </div>

                  </div>

                  {/* Disease */}

                  <div className="rounded-xl border border-[#D9E0D4] p-5">

                    <p className="text-sm text-[#5A6B58]">
                      Disease analysis
                    </p>

                    {scanResult.disease.available ? (
                      <>
                        <h4 className="mt-1 text-xl font-semibold text-[#1E3A2E]">
                          {scanResult.disease.name?.replace(
                            /_/g,
                            " "
                          )}
                        </h4>

                        {scanResult.disease.confidence !==
                          null && (
                          <p className="mt-1 text-sm text-[#5A6B58]">
                            Confidence:{" "}
                            {scanResult.disease.confidence.toFixed(
                              2
                            )}
                            %
                          </p>
                        )}

                        <div
                          className={`mt-4 inline-flex px-3 py-1.5 rounded-full text-sm font-medium ${
                            scanResult.disease.is_healthy
                              ? "bg-[#EAF0E6] text-[#1E3A2E]"
                              : "bg-[#F8E8E3] text-[#8A3D2C]"
                          }`}
                        >
                          {scanResult.disease.is_healthy
                            ? "Healthy"
                            : "Disease detected"}
                        </div>

                        {scanResult.disease.severity && (
                          <p className="mt-3 text-sm text-[#5A6B58]">
                            Severity:{" "}
                            {scanResult.disease.severity}
                          </p>
                        )}
                      </>
                    ) : (
                      <div className="mt-3">

                        <p className="text-base font-medium text-[#1E3A2E]">
                          Disease analysis not available
                        </p>

                        <p className="mt-1 text-sm text-[#5A6B58]">
                          The current disease model
                          does not support{" "}
                          <span className="font-medium">
                            {scanResult.species.name}
                          </span>
                          .
                        </p>

                      </div>
                    )}

                  </div>

                </div>
              )}

            </div>

          </div>

          {/* Error */}

          {error && (
            <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3">

              <p className="text-sm text-red-700">
                {error}
              </p>

            </div>
          )}

        </div>

      </section>

    </main>
  );
}