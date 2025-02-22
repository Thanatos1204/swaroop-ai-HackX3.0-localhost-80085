import Navbar from '@/components/shared/Navbar'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="pb-10">{children}</main>
    </div>
  )
}