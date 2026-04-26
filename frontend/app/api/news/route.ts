import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const mlUrl = process.env.ML_URL || 'http://localhost:8000'
    const res = await fetch(`${mlUrl}/news`, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
      },
      next: { revalidate: 0 },
    })

    if (!res.ok) {
      console.error('ML API news failed:', res.statusText)
      return NextResponse.json([], { status: 200 })
    }

    const data = await res.json()
    return NextResponse.json(data, { status: 200 })
  } catch (error: any) {
    console.error('Error fetching /news:', error.message)
    return NextResponse.json([], { status: 200 })
  }
}
