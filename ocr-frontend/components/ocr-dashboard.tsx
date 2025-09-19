"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ImageViewer } from "./image-viewer"
import { FormEditor } from "./form-editor"
import { SaveIcon, RefreshCwIcon } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface OCRData {
  image_path: string
  total_regions: number
  form_fields: Record<string, string>
  raw_regions: Record<
    string,
    {
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
  >
}

export function OCRDashboard() {
  const [ocrData, setOcrData] = useState<OCRData | null>(null)
  const [editedData, setEditedData] = useState<Record<string, string>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    loadOCRData()
  }, [])

  const loadOCRData = async () => {
    try {
      const response = await fetch("/data/output.json")
      const data = await response.json()
      setOcrData(data)
      setEditedData(data.form_fields)
    } catch (error) {
      console.error("Failed to load OCR data:", error)
      toast({
        title: "Error",
        description: "Failed to load OCR data",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleFieldChange = (field: string, value: string) => {
    setEditedData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const handleSaveCorrections = async () => {
    setIsSaving(true)
    try {
      // Create updated data structure
      const updatedData = {
        ...ocrData,
        form_fields: editedData,
        last_modified: new Date().toISOString(),
        manually_corrected: true,
      }

      // In a real application, this would save to your backend/file system
      // For demo purposes, we'll simulate the save and show success
      await new Promise((resolve) => setTimeout(resolve, 1000))

      console.log("Saving corrected data:", updatedData)

      toast({
        title: "Success",
        description: "Corrections saved successfully",
      })
    } catch (error) {
      console.error("Failed to save corrections:", error)
      toast({
        title: "Error",
        description: "Failed to save corrections",
        variant: "destructive",
      })
    } finally {
      setIsSaving(false)
    }
  }

  const hasChanges = JSON.stringify(editedData) !== JSON.stringify(ocrData?.form_fields)

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex items-center gap-2">
          <RefreshCwIcon className="h-4 w-4 animate-spin" />
          <span>Loading OCR data...</span>
        </div>
      </div>
    )
  }

  if (!ocrData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="w-96">
          <CardContent className="pt-6">
            <p className="text-center text-muted-foreground">No OCR data available</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">OCR Data Verification</h1>
          <p className="text-muted-foreground">Review and correct extracted text from the scanned document</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">{ocrData.total_regions} regions detected</Badge>
          {hasChanges && (
            <Button onClick={handleSaveCorrections} disabled={isSaving} className="gap-2">
              <SaveIcon className="h-4 w-4" />
              {isSaving ? "Saving..." : "Save Corrections"}
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Image Viewer */}
        <Card>
          <CardHeader>
            <CardTitle>Original Document</CardTitle>
          </CardHeader>
          <CardContent>
            <ImageViewer imageSrc="/images/debug_detection.png" regions={ocrData.raw_regions} />
          </CardContent>
        </Card>

        {/* Form Editor */}
        <Card>
          <CardHeader>
            <CardTitle>Extracted Data</CardTitle>
          </CardHeader>
          <CardContent>
            <FormEditor originalData={ocrData.form_fields} editedData={editedData} onFieldChange={handleFieldChange} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
