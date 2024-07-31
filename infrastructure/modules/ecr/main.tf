resource "aws_ecr_repository" "ecr_repo" {
  name = var.repository_name

  image_scanning_configuration {
    scan_on_push = true
  }
}
