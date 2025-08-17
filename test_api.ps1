$body = @{
    entity = "Tesla"
    category = "automotive"
    providers = @("openai")
} | ConvertTo-Json

Write-Host "Testing API with request: $body"

try {
    $response = Invoke-WebRequest -Uri "http://localhost:5051/api/visibility" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
    Write-Host "Status: $($response.StatusCode)"
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
} 