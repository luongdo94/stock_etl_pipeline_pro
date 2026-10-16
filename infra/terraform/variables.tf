variable "supabase_project_id" {
  description = "The ID of your Supabase project (found in the dashboard URL)"
  type        = string
}

variable "supabase_access_token" {
  description = "Supabase Access Token or Service Role Key"
  type        = string
  sensitive   = true
}

variable "bucket_name" {
  description = "Name of the storage bucket for the data warehouse"
  type        = string
  default     = "warehouse"
}
