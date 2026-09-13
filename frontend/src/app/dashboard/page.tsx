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
  getScanHistory,
  ScanResult,
} from "../../services/scan";

export default function DashboardPage() {
  const router = useRouter();

  // ============================================================
  // USER STATE
  // ============================================================

  const [user, setUser] = useState<User | null>(null);

  const [loadingUser, setLoadingUser] =
    useState(true);

  // ============================================================
  // IMAGE STATE
  // ============================================================

  const [selectedImage, setSelectedImage] =
    useState<File | null>(null);

  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  // ============================================================
  // SCAN RESULT STATE
  // ============================================================

  const [scanResult, setScanResult] =
    useState<ScanResult | null>(null);

  const [scanning, setScanning] =
    useState(false);

  const [error, setError] =
    useState("");

  // ============================================================
  // SCAN HISTORY STATE
  // ============================================================

  const [scanHistory, setScanHistory] =
    useState<ScanResult[]>([]);

  const [loadingHistory, setLoadingHistory] =
    useState(true);

  const [historyError, setHistoryError] =
    useState("");

  const [showAllHistory, setShowAllHistory] =
    useState(false);

  // ============================================================
  // LOAD USER + HISTORY
  // ============================================================

  useEffect(() => {
    let mounted = true;

    const loadDashboard = async () => {
      const token =
        localStorage.getItem("access_token");

      // --------------------------------------------------------
      // No token
      // --------------------------------------------------------

      if (!token) {
        router.replace("/login");
        return;
      }

      try {
        // ------------------------------------------------------
        // Get current user
        // ------------------------------------------------------

        const userData =
          await getCurrentUser();

        if (!mounted) return;

        setUser(userData.user);

        localStorage.setItem(
          "user",
          JSON.stringify(userData.user)
        );

        setLoadingUser(false);

        // ------------------------------------------------------
        // Get scan history
        // ------------------------------------------------------

        setHistoryError("");
        setLoadingHistory(true);

        const historyData =
          await getScanHistory();

        if (!mounted) return;

        setScanHistory(
          historyData.scans || []
        );
      } catch (err: any) {
        console.error(
          "Dashboard loading error:",
          err
        );

        if (!mounted) return;

        // ------------------------------------------------------
        // Token expired / invalid
        // ------------------------------------------------------

        if (
          err?.response?.status === 401
        ) {
          localStorage.removeItem(
            "access_token"
          );

          localStorage.removeItem(
            "user"
          );

          router.replace("/login");
          return;
        }

        setHistoryError(
          err?.response?.data?.error ||
            "Unable to load dashboard data."
        );
      } finally {
        if (mounted) {
          setLoadingUser(false);
          setLoadingHistory(false);
        }
      }
    };

    loadDashboard();

    return () => {
      mounted = false;
    };
  }, [router]);

  // ============================================================
  // CLEAN PREVIEW URL
  // ============================================================

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // ============================================================
  // IMAGE SELECTION
  // ============================================================

  const handleImageChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    setError("");
    setScanResult(null);

    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    // ----------------------------------------------------------
    // Allowed image types
    // ----------------------------------------------------------

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

    // ----------------------------------------------------------
    // Maximum size: 10 MB
    // ----------------------------------------------------------

    const maxSize =
      10 * 1024 * 1024;

    if (file.size > maxSize) {
      setError(
        "Image size must be less than 10 MB."
      );

      event.target.value = "";
      return;
    }

    // ----------------------------------------------------------
    // Remove old preview
    // ----------------------------------------------------------

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    // ----------------------------------------------------------
    // Create new preview
    // ----------------------------------------------------------

    const url =
      URL.createObjectURL(file);

    setSelectedImage(file);
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
      // --------------------------------------------------------
      // Send image to backend
      // --------------------------------------------------------

      const response =
        await scanPlant(selectedImage);

      const newScan =
        response.scan;

      // --------------------------------------------------------
      // Show result
      // --------------------------------------------------------

      setScanResult(newScan);

      // --------------------------------------------------------
      // Add newest scan to beginning of history
      // --------------------------------------------------------

      setScanHistory(
        (previousHistory) => [
          newScan,
          ...previousHistory.filter(
            (scan) =>
              scan.id !== newScan.id
          ),
        ]
      );
    } catch (err: any) {
      console.error(
        "Plant scan error:",
        err
      );

      // --------------------------------------------------------
      // JWT expired
      // --------------------------------------------------------

      if (
        err?.response?.status === 401
      ) {
        localStorage.removeItem(
          "access_token"
        );

        localStorage.removeItem(
          "user"
        );

        router.replace("/login");
        return;
      }

      const backendMessage =
        err?.response?.data?.error ||
        err?.response?.data?.message;

      setError(
        backendMessage ||
          "Unable to scan the plant. Please try again."
      );
    } finally {
      setScanning(false);
    }
  };

  // ============================================================
  // CLEAR CURRENT SCAN
  // ============================================================

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

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
  // REFRESH HISTORY
  // ============================================================

  const handleRefreshHistory = async () => {
    setHistoryError("");
    setLoadingHistory(true);

    try {
      const data =
        await getScanHistory();

      setScanHistory(
        data.scans || []
      );
    } catch (err: any) {
      console.error(
        "Refresh history error:",
        err
      );

      if (
        err?.response?.status === 401
      ) {
        localStorage.removeItem(
          "access_token"
        );

        localStorage.removeItem(
          "user"
        );

        router.replace("/login");
        return;
      }

      setHistoryError(
        err?.response?.data?.error ||
          "Unable to refresh scan history."
      );
    } finally {
      setLoadingHistory(false);
    }
  };

  // ============================================================
  // SELECT HISTORY RESULT
  // ============================================================

  const handleHistorySelect = (
    scan: ScanResult
  ) => {
    setScanResult(scan);
    setError("");
    setShowAllHistory(false);

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  // ============================================================
  // LOGOUT
  // ============================================================

  const handleLogout = () => {
    localStorage.removeItem(
      "access_token"
    );

    localStorage.removeItem(
      "user"
    );

    router.replace("/login");
  };

  // ============================================================
  // FORMAT DATE
  // ============================================================

  const formatDate = (
    dateString?: string | null
  ) => {
    if (!dateString) {
      return "Date unavailable";
    }

    const date =
      new Date(dateString);

    if (Number.isNaN(date.getTime())) {
      return "Date unavailable";
    }

    return date.toLocaleString(
      "en-IN",
      {
        dateStyle: "medium",
        timeStyle: "short",
      }
    );
  };

  // ============================================================
  // LOADING
  // ============================================================

  if (loadingUser) {
    return (
      <main className="min-h-screen flex items-center justify-center bg-[#F7F5EF]">

        <div className="text-center">

          <div className="text-4xl animate-pulse">
            🌱
          </div>

          <p className="mt-4 text-[#5A6B58]">
            Loading your dashboard...
          </p>

        </div>

      </main>
    );
  }

  // ============================================================
  // NO USER
  // ============================================================

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

        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">

          <div>

            <h1 className="text-2xl font-semibold text-[#1E3A2E]">
              BotoniiQ
            </h1>

            <p className="text-sm text-[#5A6B58]">
              Intelligent plant health assistant
            </p>

          </div>

          <button
            type="button"
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

      <section className="max-w-7xl mx-auto px-6 py-10">

        {/* ====================================================
            WELCOME
        ==================================================== */}

        <div className="mb-8">

          <p className="text-sm text-[#5A6B58]">
            Welcome back
          </p>

          <h2 className="mt-1 text-3xl font-semibold text-[#14201A]">
            {user.name}
          </h2>

          <p className="mt-2 text-[#5A6B58]">
            Upload a plant image to identify its
            species and check available disease
            information.
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
              Upload a clear image of the plant
              leaf for AI-powered analysis.
            </p>

          </div>

          {/* ==================================================
              IMAGE + RESULT
          ================================================== */}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

            {/* =================================================
                UPLOAD
            ================================================= */}

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
                onChange={
                  handleImageChange
                }
                className="hidden"
              />

              {/* Selected file */}

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

            {/* =================================================
                RESULT
            ================================================= */}

            <div>

              {/* Empty */}

              {!scanResult &&
                !scanning && (
                  <div className="min-h-[280px] rounded-xl border border-[#D9E0D4] bg-[#FAFAF7] flex flex-col items-center justify-center text-center p-6">

                    <div className="text-4xl">
                      🔍
                    </div>

                    <p className="mt-4 font-medium text-[#1E3A2E]">
                      Scan result
                    </p>

                    <p className="mt-2 text-sm text-[#5A6B58] max-w-sm">
                      Your plant identification
                      and available disease
                      analysis will appear here.
                    </p>

                  </div>
                )}

              {/* Scanning */}

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

              {/* Result */}

              {scanResult && (
                <div className="space-y-4">

                  {/* =================================================
                      SPECIES
                  ================================================= */}

                  <div className="rounded-xl border border-[#D9E0D4] p-5">

                    <p className="text-sm text-[#5A6B58]">
                      Identified species
                    </p>

                    <div className="mt-1 flex items-center justify-between gap-4">

                      <h4 className="text-2xl font-semibold text-[#1E3A2E]">
                        {
                          scanResult
                            .species
                            .name
                        }
                      </h4>

                      <span className="text-sm font-semibold text-[#C98A3B]">
                        {
                          scanResult
                            .species
                            .confidence
                            .toFixed(2)
                        }
                        %
                      </span>

                    </div>

                  </div>

                  {/* =================================================
                      TOP 3
                  ================================================= */}

                  <div className="rounded-xl border border-[#D9E0D4] p-5">

                    <p className="text-sm font-medium text-[#14201A]">
                      Top predictions
                    </p>

                    <div className="mt-3 space-y-3">

                      {scanResult.species.top_3.map(
                        (
                          prediction,
                          index
                        ) => (
                          <div
                            key={`${prediction.name}-${index}`}
                            className="flex items-center justify-between text-sm"
                          >

                            <span className="text-[#5A6B58]">
                              {index + 1}.
                              {" "}
                              {
                                prediction.name
                              }
                            </span>

                            <span className="font-medium text-[#1E3A2E]">
                              {
                                prediction
                                  .confidence
                                  .toFixed(2)
                              }
                              %
                            </span>

                          </div>
                        )
                      )}

                    </div>

                  </div>

                  {/* =================================================
                      DISEASE
                  ================================================= */}

                  <div className="rounded-xl border border-[#D9E0D4] p-5">

                    <p className="text-sm text-[#5A6B58]">
                      Disease analysis
                    </p>

                    {scanResult.disease.available ? (
                      <>

                        <h4 className="mt-1 text-xl font-semibold text-[#1E3A2E]">
                          {
                            scanResult
                              .disease
                              .name
                              ?.replace(
                                /_/g,
                                " "
                              )
                          }
                        </h4>

                        {scanResult
                          .disease
                          .confidence !==
                          null && (
                          <p className="mt-1 text-sm text-[#5A6B58]">
                            Confidence:{" "}
                            {
                              scanResult
                                .disease
                                .confidence
                                .toFixed(2)
                            }
                            %
                          </p>
                        )}

                        <div
                          className={`mt-4 inline-flex px-3 py-1.5 rounded-full text-sm font-medium ${
                            scanResult
                              .disease
                              .is_healthy
                              ? "bg-[#EAF0E6] text-[#1E3A2E]"
                              : "bg-[#F8E8E3] text-[#8A3D2C]"
                          }`}
                        >
                          {scanResult
                            .disease
                            .is_healthy
                            ? "Healthy"
                            : "Disease detected"}
                        </div>

                        {scanResult
                          .disease
                          .severity && (
                          <p className="mt-3 text-sm text-[#5A6B58]">
                            Severity:{" "}
                            {
                              scanResult
                                .disease
                                .severity
                            }
                          </p>
                        )}

                      </>
                    ) : (
                      <div className="mt-3">

                        <p className="text-base font-medium text-[#1E3A2E]">
                          Disease analysis not available
                        </p>

                        <p className="mt-1 text-sm text-[#5A6B58]">
                          The current disease
                          model does not support{" "}
                          <span className="font-medium">
                            {
                              scanResult
                                .species
                                .name
                            }
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

          {/* ==================================================
              ERROR
          ================================================== */}

          {error && (
            <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3">

              <p className="text-sm text-red-700">
                {error}
              </p>

            </div>
          )}

        </div>

        {/* ======================================================
            RECENT SCANS
        ====================================================== */}

        <div className="mt-8 bg-white rounded-2xl border border-[#D9E0D4] p-6">

          {/* ====================================================
              HISTORY HEADER
          ==================================================== */}

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">

            <div>

              <h3 className="text-xl font-semibold text-[#14201A]">
                Recent scans
              </h3>

              <p className="mt-1 text-sm text-[#5A6B58]">
                Your latest plant identification results
              </p>

            </div>

            <div className="flex items-center gap-3">

              {/* Total count */}

              <div className="px-3 py-2 rounded-lg bg-[#F3F5EF]">

                <span className="text-sm text-[#5A6B58]">
                  Total scans:{" "}
                </span>

                <span className="text-sm font-semibold text-[#1E3A2E]">
                  {scanHistory.length}
                </span>

              </div>

              {/* Refresh */}

              <button
                type="button"
                onClick={
                  handleRefreshHistory
                }
                disabled={loadingHistory}
                className="px-4 py-2 rounded-lg border border-[#D9E0D4] text-sm text-[#1E3A2E] hover:bg-[#F3F5EF] transition-colors disabled:opacity-50"
              >
                {loadingHistory
                  ? "Refreshing..."
                  : "Refresh"}
              </button>

            </div>

          </div>

          {/* ====================================================
              HISTORY ERROR
          ==================================================== */}

          {historyError && (
            <div className="mt-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3">

              <p className="text-sm text-red-700">
                {historyError}
              </p>

            </div>
          )}

          {/* ====================================================
              HISTORY LOADING
          ==================================================== */}

          {loadingHistory && (
            <div className="mt-6 py-10 text-center">

              <div className="text-3xl animate-pulse">
                🌱
              </div>

              <p className="mt-3 text-sm text-[#5A6B58]">
                Loading your scans...
              </p>

            </div>
          )}

          {/* ====================================================
              EMPTY HISTORY
          ==================================================== */}

          {!loadingHistory &&
            !historyError &&
            scanHistory.length === 0 && (
              <div className="mt-6 rounded-xl border border-dashed border-[#D9E0D4] bg-[#FAFAF7] py-12 text-center">

                <div className="text-4xl">
                  🌿
                </div>

                <p className="mt-4 font-medium text-[#1E3A2E]">
                  No scans yet
                </p>

                <p className="mt-1 text-sm text-[#5A6B58]">
                  Your plant scans will appear
                  here after your first scan.
                </p>

              </div>
            )}

          {/* ====================================================
              RECENT 3 SCANS
          ==================================================== */}

          {!loadingHistory &&
            scanHistory.length > 0 && (
              <>

                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">

                  {scanHistory
                    .slice(0, 3)
                    .map(
                      (
                        scan,
                        index
                      ) => (
                        <button
                          key={`${scan.id}-${index}`}
                          type="button"
                          onClick={() =>
                            handleHistorySelect(
                              scan
                            )
                          }
                          className="group text-left rounded-xl border border-[#D9E0D4] p-5 hover:border-[#C98A3B] hover:shadow-sm bg-[#FEFEFC] transition-all"
                        >

                          {/* ==================================
                              TOP
                          ================================== */}

                          <div className="flex items-start justify-between gap-3">

                            <div className="flex items-center gap-3 min-w-0">

                              <div className="w-10 h-10 shrink-0 rounded-full bg-[#EAF0E6] flex items-center justify-center text-xl">
                                🌿
                              </div>

                              <div className="min-w-0">

                                <p className="font-semibold text-[#1E3A2E] truncate">
                                  {
                                    scan
                                      .species
                                      .name
                                  }
                                </p>

                                <p className="mt-1 text-xs text-[#8A9584]">
                                  Scan #{scan.id}
                                </p>

                              </div>

                            </div>

                            <span className="text-xs text-[#8A9584] whitespace-nowrap">
                              {
                                formatDate(
                                  scan.created_at
                                ).split(",")[0]
                              }
                            </span>

                          </div>

                          {/* ==================================
                              CONFIDENCE
                          ================================== */}

                          <div className="mt-5">

                            <div className="flex items-center justify-between">

                              <span className="text-sm text-[#5A6B58]">
                                Confidence
                              </span>

                              <span className="text-sm font-semibold text-[#1E3A2E]">
                                {
                                  scan
                                    .species
                                    .confidence
                                    .toFixed(2)
                                }
                                %
                              </span>

                            </div>

                            <div className="mt-2 h-2 rounded-full bg-[#EAF0E6] overflow-hidden">

                              <div
                                className="h-full rounded-full bg-[#8EA69B]"
                                style={{
                                  width: `${Math.min(
                                    scan
                                      .species
                                      .confidence,
                                    100
                                  )}%`,
                                }}
                              />

                            </div>

                          </div>

                          {/* ==================================
                              DISEASE STATUS
                          ================================== */}

                          <div className="mt-5">

                            {scan.disease.available ? (
                              <div className="flex items-center justify-between gap-3">

                                <span
                                  className={`inline-flex px-3 py-1.5 rounded-full text-xs font-medium ${
                                    scan
                                      .disease
                                      .is_healthy
                                      ? "bg-[#EAF0E6] text-[#1E3A2E]"
                                      : "bg-[#F8E8E3] text-[#8A3D2C]"
                                  }`}
                                >
                                  {scan
                                    .disease
                                    .is_healthy
                                    ? "Healthy"
                                    : "Disease detected"}
                                </span>

                                {!scan
                                  .disease
                                  .is_healthy &&
                                  scan.disease
                                    .name && (
                                    <span className="text-xs text-[#5A6B58] truncate">
                                      {
                                        scan
                                          .disease
                                          .name
                                          .replace(
                                            /_/g,
                                            " "
                                          )
                                      }
                                    </span>
                                  )}

                              </div>
                            ) : (
                              <span className="inline-flex px-3 py-1.5 rounded-full text-xs font-medium bg-[#F3F5EF] text-[#5A6B58]">
                                Disease analysis unavailable
                              </span>
                            )}

                          </div>

                          {/* ==================================
                              VIEW RESULT
                          ================================== */}

                          <div className="mt-5 pt-4 border-t border-[#EEF1EB]">

                            <span className="text-xs font-medium text-[#C98A3B] group-hover:underline">
                              View result →
                            </span>

                          </div>

                        </button>
                      )
                    )}

                </div>

                {/* ==================================================
                    VIEW ALL BUTTON
                ================================================== */}

                {scanHistory.length > 3 && (
                  <div className="mt-6 flex justify-center">

                    <button
                      type="button"
                      onClick={() =>
                        setShowAllHistory(
                          true
                        )
                      }
                      className="px-5 py-2.5 rounded-lg border border-[#D9E0D4] text-sm font-medium text-[#1E3A2E] hover:bg-[#F3F5EF] transition-colors"
                    >
                      View all{" "}
                      {scanHistory.length}{" "}
                      scans
                    </button>

                  </div>
                )}

              </>
            )}

        </div>

      </section>

      {/* ========================================================
          ALL SCANS MODAL
      ======================================================== */}

      {showAllHistory && (
        <div
          className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4"
          onClick={() =>
            setShowAllHistory(false)
          }
        >

          <div
            className="w-full max-w-4xl max-h-[85vh] bg-white rounded-2xl shadow-xl overflow-hidden"
            onClick={(event) =>
              event.stopPropagation()
            }
          >

            {/* ==================================================
                MODAL HEADER
            ================================================== */}

            <div className="px-6 py-5 border-b border-[#D9E0D4] flex items-center justify-between">

              <div>

                <h3 className="text-xl font-semibold text-[#14201A]">
                  Scan history
                </h3>

                <p className="mt-1 text-sm text-[#5A6B58]">
                  {scanHistory.length} total scans
                </p>

              </div>

              <button
                type="button"
                onClick={() =>
                  setShowAllHistory(
                    false
                  )
                }
                className="w-9 h-9 rounded-full border border-[#D9E0D4] text-[#5A6B58] hover:bg-[#F3F5EF] transition-colors"
                aria-label="Close scan history"
              >
                ✕
              </button>

            </div>

            {/* ==================================================
                MODAL CONTENT
            ================================================== */}

            <div className="p-6 overflow-y-auto max-h-[calc(85vh-90px)]">

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                {scanHistory.map(
                  (
                    scan,
                    index
                  ) => (
                    <button
                      key={`${scan.id}-${index}`}
                      type="button"
                      onClick={() =>
                        handleHistorySelect(
                          scan
                        )
                      }
                      className="text-left rounded-xl border border-[#D9E0D4] p-5 hover:border-[#C98A3B] hover:bg-[#FAFAF7] transition-colors"
                    >

                      {/* --------------------------------------
                          MODAL CARD TOP
                      -------------------------------------- */}

                      <div className="flex items-start justify-between gap-3">

                        <div className="flex items-center gap-3">

                          <div className="w-10 h-10 rounded-full bg-[#EAF0E6] flex items-center justify-center text-xl">
                            🌿
                          </div>

                          <div>

                            <p className="font-semibold text-[#1E3A2E]">
                              {
                                scan
                                  .species
                                  .name
                              }
                            </p>

                            <p className="mt-1 text-xs text-[#8A9584]">
                              Scan #{scan.id}
                            </p>

                          </div>

                        </div>

                        <span className="text-xs text-[#8A9584] text-right">
                          {
                            formatDate(
                              scan.created_at
                            )
                          }
                        </span>

                      </div>

                      {/* --------------------------------------
                          CONFIDENCE
                      -------------------------------------- */}

                      <div className="mt-4 flex items-center justify-between">

                        <span className="text-sm text-[#5A6B58]">
                          Confidence
                        </span>

                        <span className="text-sm font-semibold text-[#1E3A2E]">
                          {
                            scan
                              .species
                              .confidence
                              .toFixed(2)
                          }
                          %
                        </span>

                      </div>

                      {/* --------------------------------------
                          DISEASE
                      -------------------------------------- */}

                      <div className="mt-3">

                        {scan.disease.available ? (
                          <span
                            className={`inline-flex px-3 py-1.5 rounded-full text-xs font-medium ${
                              scan
                                .disease
                                .is_healthy
                                ? "bg-[#EAF0E6] text-[#1E3A2E]"
                                : "bg-[#F8E8E3] text-[#8A3D2C]"
                            }`}
                          >
                            {scan
                              .disease
                              .is_healthy
                              ? "Healthy"
                              : scan
                                  .disease
                                  .name
                                ?.replace(
                                  /_/g,
                                  " "
                                )}
                          </span>
                        ) : (
                          <span className="inline-flex px-3 py-1.5 rounded-full text-xs font-medium bg-[#F3F5EF] text-[#5A6B58]">
                            Disease analysis unavailable
                          </span>
                        )}

                      </div>

                      {/* --------------------------------------
                          VIEW
                      -------------------------------------- */}

                      <div className="mt-4 pt-3 border-t border-[#EEF1EB]">

                        <span className="text-xs font-medium text-[#C98A3B]">
                          View result →
                        </span>

                      </div>

                    </button>
                  )
                )}

              </div>

            </div>

          </div>

        </div>
      )}

    </main>
  );
}