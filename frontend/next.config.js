/** @type {import('next').NextConfig} */
const nextConfig = {}
const required = ['NEXT_PUBLIC_SUPABASE_URL', 'NEXT_PUBLIC_SUPABASE_ANON_KEY']
if (process.env.NODE_ENV === 'production') {
  required.forEach((key) => {
    if (!process.env[key]) throw new Error(`Missing env var: ${key}`)
  })
}
module.exports = nextConfig
