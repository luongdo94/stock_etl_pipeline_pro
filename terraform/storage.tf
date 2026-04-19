resource "supabase_storage_bucket" "warehouse" {
  project_ref   = var.supabase_project_id
  name          = var.bucket_name
  public        = false
  file_size_limit = 52428800 # 50MB
  allowed_mime_types = ["application/x-parquet", "application/octet-stream"]
}

# (Tùy chọn) Thêm Row Level Security (RLS) Policy cho Storage nếu cần
# resource "supabase_storage_policy" "api_access" { ... }
