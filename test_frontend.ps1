Write-Host "🧪 Testing LLM Visibility Frontend & API Connection" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Test 1: Frontend accessibility
Write-Host "`n1️⃣ Testing Frontend (Port 8080)..." -ForegroundColor Yellow
try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing
    Write-Host "   ✅ Frontend accessible: Status $($frontendResponse.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Frontend error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: API accessibility
Write-Host "`n2️⃣ Testing API (Port 5051)..." -ForegroundColor Yellow
try {
    $apiResponse = Invoke-WebRequest -Uri "http://localhost:5051/health" -UseBasicParsing
    Write-Host "   ✅ API accessible: Status $($apiResponse.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ API error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: API functionality with real request
Write-Host "`n3️⃣ Testing API Functionality..." -ForegroundColor Yellow
try {
    $testBody = @{
        entity = "Apple"
        category = "technology"
        providers = @("openai")
    } | ConvertTo-Json

    Write-Host "   📤 Sending test request for 'Apple'..." -ForegroundColor Cyan
    
    $apiTestResponse = Invoke-WebRequest -Uri "http://localhost:5051/api/visibility" -Method POST -Body $testBody -ContentType "application/json" -UseBasicParsing
    
    if ($apiTestResponse.StatusCode -eq 200) {
        $data = $apiTestResponse.Content | ConvertFrom-Json
        Write-Host "   ✅ API test successful!" -ForegroundColor Green
        Write-Host "   📊 Entity: $($data.entity)" -ForegroundColor White
        Write-Host "   🎯 Overall Score: $($data.overall)/100" -ForegroundColor White
        Write-Host "   🔍 Recognition: $([math]::Round($data.subscores.recognition * 10, 1))/10" -ForegroundColor White
        Write-Host "   📝 Detail: $([math]::Round($data.subscores.detail * 10, 1))/10" -ForegroundColor White
    } else {
        Write-Host "   ❌ API test failed: Status $($apiTestResponse.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "   ❌ API test error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Frontend-API integration
Write-Host "`n4️⃣ Testing Frontend-API Integration..." -ForegroundColor Yellow
try {
    # Simulate what the frontend would send
    $frontendTestBody = @{
        entity = "Tesla"
        category = "automotive"
        providers = @("openai")
    } | ConvertTo-Json

    Write-Host "   📤 Testing frontend-style request for 'Tesla'..." -ForegroundColor Cyan
    
    $integrationResponse = Invoke-WebRequest -Uri "http://localhost:5051/api/visibility" -Method POST -Body $frontendTestBody -ContentType "application/json" -UseBasicParsing
    
    if ($integrationResponse.StatusCode -eq 200) {
        $integrationData = $integrationResponse.Content | ConvertFrom-Json
        Write-Host "   ✅ Integration test successful!" -ForegroundColor Green
        Write-Host "   🚗 Entity: $($integrationData.entity)" -ForegroundColor White
        Write-Host "   🎯 Score: $($integrationData.overall)/100" -ForegroundColor White
        Write-Host "   🔌 Provider: $($integrationData.providers[0].provider)" -ForegroundColor White
        Write-Host "   🤖 Model: $($integrationData.providers[0].model)" -ForegroundColor White
    }
} catch {
    Write-Host "   ❌ Integration test error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🎉 Testing Complete!" -ForegroundColor Green
Write-Host "🌐 Frontend: http://localhost:8080" -ForegroundColor Cyan
Write-Host "🔗 API: http://localhost:5051" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:5051/docs" -ForegroundColor Cyan 