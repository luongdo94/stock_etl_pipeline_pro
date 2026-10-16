terraform {
  required_providers {
    supabase = {
      source  = "supabase/supabase"
      version = "~> 1.0"
    }
  }
}

provider "supabase" {
  # Cấu hình thông qua biến môi trường hoặc file .tfvars
  # API_KEY nên là Access Token cá nhân hoặc Service Role Key tùy mục đích
}
