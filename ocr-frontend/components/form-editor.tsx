"use client"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertCircleIcon, CheckCircleIcon } from "lucide-react"

interface FormEditorProps {
  originalData: Record<string, string>
  editedData: Record<string, string>
  onFieldChange: (field: string, value: string) => void
}

export function FormEditor({ originalData, editedData, onFieldChange }: FormEditorProps) {
  const fieldEntries = Object.entries(originalData)

  const getFieldStatus = (field: string) => {
    const original = originalData[field]
    const edited = editedData[field]

    if (edited !== original) {
      return "modified"
    }

    // Check if field might need attention (common OCR issues)
    if (original.length < 2 || /[^a-zA-Z0-9\s@.-]/.test(original)) {
      return "warning"
    }

    return "good"
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "modified":
        return <CheckCircleIcon className="h-4 w-4 text-blue-500" />
      case "warning":
        return <AlertCircleIcon className="h-4 w-4 text-yellow-500" />
      default:
        return <CheckCircleIcon className="h-4 w-4 text-green-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "modified":
        return (
          <Badge variant="default" className="text-xs">
            Modified
          </Badge>
        )
      case "warning":
        return (
          <Badge variant="secondary" className="text-xs">
            Check
          </Badge>
        )
      default:
        return null
    }
  }

  return (
    <div className="space-y-4 max-h-96 overflow-y-auto">
      {fieldEntries.map(([field, originalValue]) => {
        const status = getFieldStatus(field)
        const currentValue = editedData[field] || ""

        return (
          <Card key={field} className="p-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor={field} className="font-medium">
                  {field}
                </Label>
                <div className="flex items-center gap-2">
                  {getStatusIcon(status)}
                  {getStatusBadge(status)}
                </div>
              </div>

              <Input
                id={field}
                value={currentValue}
                onChange={(e) => onFieldChange(field, e.target.value)}
                placeholder={`Enter ${field.toLowerCase()}`}
                className={status === "warning" ? "border-yellow-300" : ""}
              />

              {originalValue !== currentValue && (
                <div className="text-xs text-muted-foreground">Original: "{originalValue}"</div>
              )}
            </div>
          </Card>
        )
      })}
    </div>
  )
}
