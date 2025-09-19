"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ZoomInIcon, ZoomOutIcon, RotateCcwIcon } from "lucide-react"

interface Region {
  value: string
  confidence: number
  bbox: number[]
  position: {
    x: number
    y: number
    width: number
    height: number
  }
}

interface ImageViewerProps {
  imageSrc: string
  regions: Record<string, Region>
}

export function ImageViewer({ imageSrc, regions }: ImageViewerProps) {
  const [zoom, setZoom] = useState(1)
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null)

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.2, 3))
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.2, 0.5))
  const handleReset = () => {
    setZoom(1)
    setSelectedRegion(null)
  }

  const colors = [
    "rgb(239, 68, 68)", // red
    "rgb(59, 130, 246)", // blue
    "rgb(34, 197, 94)", // green
    "rgb(249, 115, 22)", // orange
    "rgb(168, 85, 247)", // purple
    "rgb(139, 69, 19)", // brown
    "rgb(236, 72, 153)", // pink
    "rgb(107, 114, 128)", // gray
    "rgb(132, 204, 22)", // lime
    "rgb(30, 64, 175)", // navy
    "rgb(6, 182, 212)", // cyan
    "rgb(217, 70, 239)", // magenta
  ]

  const regionEntries = Object.entries(regions)

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={handleZoomOut}>
          <ZoomOutIcon className="h-4 w-4" />
        </Button>
        <span className="text-sm text-muted-foreground min-w-16 text-center">{Math.round(zoom * 100)}%</span>
        <Button variant="outline" size="sm" onClick={handleZoomIn}>
          <ZoomInIcon className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="sm" onClick={handleReset}>
          <RotateCcwIcon className="h-4 w-4" />
        </Button>
      </div>

      {/* Image Container */}
      <div className="relative overflow-auto max-h-96 border rounded-lg">
        <div className="relative inline-block" style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}>
          <img src={imageSrc || "/placeholder.svg"} alt="OCR Document" className="block max-w-none" />

          {/* Region Overlays */}
          {regionEntries.map(([regionId, region], index) => {
            const color = colors[index % colors.length]
            const isSelected = selectedRegion === regionId

            return (
              <div
                key={regionId}
                className={`absolute border-2 cursor-pointer transition-all ${
                  isSelected ? "bg-black/20" : "hover:bg-black/10"
                }`}
                style={{
                  left: region.position.x,
                  top: region.position.y,
                  width: region.position.width,
                  height: region.position.height,
                  borderColor: color,
                  borderWidth: isSelected ? 3 : 2,
                }}
                onClick={() => setSelectedRegion(isSelected ? null : regionId)}
                title={`${regionId}: "${region.value}" (${Math.round(region.confidence * 100)}%)`}
              >
                {/* Region Label */}
                <div
                  className="absolute -top-6 left-0 text-xs font-medium px-1 rounded text-white"
                  style={{ backgroundColor: color }}
                >
                  {regionId.replace("text_field_", "")}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Selected Region Info */}
      {selectedRegion && (
        <Card className="p-3">
          <div className="space-y-1">
            <div className="font-medium">{selectedRegion}</div>
            <div className="text-sm text-muted-foreground">Text: "{regions[selectedRegion].value}"</div>
            <div className="text-sm text-muted-foreground">
              Confidence: {Math.round(regions[selectedRegion].confidence * 100)}%
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
