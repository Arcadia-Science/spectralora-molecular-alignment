import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { Button } from '@/components/ui/button'
import { Input }  from '@/components/ui/input'
import { Card }   from '@/components/ui/card'

function Toggle({ checked, onChange }) {
  return (
    <div
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      style={{
        position: 'relative',
        width: 48,
        height: 26,
        borderRadius: 13,
        backgroundColor: checked ? '#58a6ff' : '#21262d',
        border: '1px solid #30363d',
        cursor: 'pointer',
        flexShrink: 0,
        transition: 'background-color 0.2s',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 3,
          left: checked ? 23 : 3,
          width: 18,
          height: 18,
          borderRadius: '50%',
          backgroundColor: 'white',
          boxShadow: '0 1px 4px rgba(0,0,0,0.5)',
          transition: 'left 0.2s',
        }}
      />
    </div>
  )
}

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 text-muted-foreground">{label} cm⁻¹</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(3)}
        </p>
      ))}
    </div>
  )
}

export default function App() {
  const [smiles,  setSmiles]  = useState('c1ccccc1')
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)
  const [showRaw, setShowRaw] = useState(false)

  const predict = async () => {
    if (!smiles.trim() || loading) return
    setLoading(true)
    setError(null)
    try {
      const r = await fetch(`${API}/predict/raman`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: smiles.trim() }),
      })
      if (!r.ok) {
        const e = await r.json().catch(() => ({ detail: r.statusText }))
        throw new Error(e.detail ?? r.statusText)
      }
      setResult(await r.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const chartData = result
    ? result.x_grid.filter((_, i) => i % 5 === 0).map((x, i) => ({
        x:       Math.round(x),
        Refined: parseFloat(result.spectrum_refined[i * 5].toFixed(4)),
        Raw:     parseFloat(result.spectrum_raw[i * 5].toFixed(4)),
      }))
    : []

  const peaks = result?.peaks.positions_cm ?? []

  return (
    /* Full-page wrapper — 32px border on all sides */
    <div className="min-h-screen p-8 font-mono">

      {/* Centered title */}
      <h1 className="mb-8 text-center text-sm font-semibold tracking-widest text-primary">
        SPECTRALORA — RAMAN
      </h1>

      {/* 75% centred column */}
      <div className="mx-auto w-3/4 space-y-4">

        {/* Input card */}
        <Card className="p-4">
          <div className="flex gap-3">
            <Input
              value={smiles}
              onChange={e => setSmiles(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && predict()}
              placeholder="SMILES — e.g. c1ccccc1"
            />
            <Button onClick={predict} disabled={loading} className="shrink-0">
              {loading ? 'running…' : 'Predict'}
            </Button>
          </div>
        </Card>

        {error && <p className="text-xs text-destructive">{error}</p>}

        {result && (
          <>
            {/* Chart card */}
            <Card className="p-6">

              {/* Title row — centered, padded from card edge */}
              <p className="mb-5 text-center text-xs font-medium uppercase tracking-widest text-muted-foreground">
                Raman Spectrum
              </p>

              {/* Chart — 55vh, padded inside */}
              <div className="px-2" style={{ height: '55vh' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 12, right: 32, bottom: 36, left: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(216 12% 12%)" />
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={[500, 4000]}
                      tickCount={8}
                      tick={{ fill: 'hsl(215 16% 47%)', fontSize: 11 }}
                      label={{ value: 'Wavenumber (cm⁻¹)', position: 'insideBottom', offset: -22, fill: 'hsl(215 16% 47%)', fontSize: 11 }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tickCount={3}
                      width={44}
                      tick={{ fill: 'hsl(215 16% 47%)', fontSize: 11 }}
                      label={{ value: 'Intensity (a.u.)', angle: -90, position: 'insideLeft', offset: 14, fill: 'hsl(215 16% 47%)', fontSize: 11 }}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    {peaks.slice(0, 8).map((p, i) => (
                      <ReferenceLine
                        key={i} x={Math.round(p)}
                        stroke="hsl(30 95% 60%)" strokeWidth={1}
                        strokeDasharray="2 5" strokeOpacity={0.35}
                      />
                    ))}
                    {showRaw && (
                      <Line type="monotone" dataKey="Raw" name="raw"
                        stroke="hsl(142 71% 45%)" strokeWidth={1}
                        strokeDasharray="4 3" dot={false} strokeOpacity={0.55}
                      />
                    )}
                    <Line type="monotone" dataKey="Refined" name="refined"
                      stroke="hsl(213 94% 68%)" strokeWidth={1.5} dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Below-chart — centered, padded from card edge */}
              <div className="mx-auto mt-5 flex max-w-sm items-center justify-between border-t border-border px-4 pt-4">
                <div className="flex items-center gap-4">
                  <span className="inline-block h-0.5 w-6 rounded" style={{ background: '#58a6ff' }} />
                  {showRaw && (
                    <span className="inline-block w-6" style={{ borderTop: '1.5px dashed hsl(142,71%,45%)' }} />
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Toggle checked={showRaw} onChange={setShowRaw} />
                  <span className="text-xs text-muted-foreground">
                    {showRaw ? 'raw' : 'refined'}
                  </span>
                </div>
              </div>
            </Card>

            {/* Meta — inside a card, centered, padded */}
            <Card className="px-6 py-4">
              <div className="flex flex-wrap justify-center gap-x-8 gap-y-2 text-xs text-muted-foreground">
                <span>{result.smiles.length > 48 ? result.smiles.slice(0, 48) + '…' : result.smiles}</span>
                <span>{result.n_atoms} atoms · {result.n_modes} modes</span>
                <span>{result.timing.total_s}s · gnn {result.timing.gnn_s}s · refnet {result.timing.refine_s}s</span>
              </div>
            </Card>

            {/* Peaks */}
            <Card className="px-6 py-5">
              <p className="mb-4 text-center text-xs font-medium uppercase tracking-widest text-muted-foreground">
                Top peaks
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {peaks.slice(0, 12).map((p, i) => (
                  <span key={i} className="rounded border border-border bg-muted px-3 py-1 text-xs text-foreground">
                    {Math.round(p)} cm⁻¹
                  </span>
                ))}
              </div>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
