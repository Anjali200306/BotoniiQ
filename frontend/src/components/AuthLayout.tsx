export default function AuthLayout({
  children,
  title,
  subtitle,
}: {
  children: React.ReactNode;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="min-h-screen flex bg-[#F3F6EE]">
      {/* Left panel — signature leaf-vein pattern */}
      <div className="hidden md:flex md:w-1/2 bg-[#1E3A2E] relative overflow-hidden items-center justify-center">
        <svg
          viewBox="0 0 400 600"
          className="absolute inset-0 w-full h-full opacity-20"
          preserveAspectRatio="xMidYMid slice"
        >
          <path
            d="M200 0 V600"
            stroke="#8FA888"
            strokeWidth="2"
            fill="none"
          />
          {Array.from({ length: 14 }).map((_, i) => {
            const y = 40 + i * 40;
            const len = 60 + (i % 3) * 20;
            return (
              <g key={i}>
                <path
                  d={`M200 ${y} Q${200 + len} ${y - 20} ${200 + len + 20} ${y + 10}`}
                  stroke="#8FA888"
                  strokeWidth="1.5"
                  fill="none"
                />
                <path
                  d={`M200 ${y} Q${200 - len} ${y - 20} ${200 - len - 20} ${y + 10}`}
                  stroke="#8FA888"
                  strokeWidth="1.5"
                  fill="none"
                />
              </g>
            );
          })}
        </svg>
        <div className="relative z-10 text-center px-12">
          <h1 className="font-[family-name:var(--font-fraunces)] text-4xl text-[#F3F6EE] font-semibold mb-3">
            BotaniQ
          </h1>
          <p className="text-[#8FA888] font-[family-name:var(--font-inter)] text-sm leading-relaxed">
            From a single photo to a complete, evolving care plan.
          </p>
        </div>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <h2 className="font-[family-name:var(--font-fraunces)] text-3xl text-[#14201A] font-semibold mb-2">
            {title}
          </h2>
          <p className="font-[family-name:var(--font-inter)] text-[#5A6B58] text-sm mb-8">
            {subtitle}
          </p>
          {children}
        </div>
      </div>
    </div>
  );
}