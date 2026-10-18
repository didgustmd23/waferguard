# ==========================================
# 승인된 원본의 최초 구조 검사 실행
# - Sandbox 계정·출력 경로·명시적 신뢰 승인 확인
# - 입력이 하나일 때만 자동 선택, 여러 개이면 사용자가 지정
# - 상세 결과는 로컬 출력 폴더에만 보존하며 원본은 수정하지 않음
# ==========================================
param([switch]$TrustedPickle, [string]$InputFile, [switch]$Canonical)
$ErrorActionPreference = 'Stop'
if ($env:USERNAME -ne 'WDAGUtilityAccount') { throw 'Sandbox 내부에서 실행하세요.' }
if (-not $TrustedPickle) { throw '출처를 확인한 뒤 -TrustedPickle로 명시적으로 승인하세요.' }
if (-not (Test-Path -LiteralPath 'C:\wm_output')) {
    throw '출력 공유 폴더가 없습니다. 템플릿을 수정한 sandbox_inspect.local.wsb로 실행하세요.'
}

# 여러 원본 중 하나를 임의로 고르거나 압축 파일을 자동 실행하지 않는다.
if (-not $InputFile) {
    $candidates = @(Get-ChildItem -LiteralPath 'C:\wm_input' -File |
        Where-Object { $_.Extension -in @('.pkl', '.pickle') })
    if ($candidates.Count -ne 1) {
        throw '입력 폴더 바로 아래의 pickle이 하나가 아닙니다. -InputFile로 지정하세요.'
    }
    $InputFile = $candidates[0].FullName
}

$sandboxPython = 'C:\wm_python\python.exe'
if (Test-Path -LiteralPath 'C:\wm_portable\active_python.txt') {
    # 준비 스크립트에서 import 검증을 통과한 독립 실행 환경을 우선한다.
    $portablePython = [IO.File]::ReadAllText('C:\wm_portable\active_python.txt').Trim()
    if ($portablePython -notmatch '^C:\\wm_portable\\[a-f0-9]{32}\\python\.exe$') {
        throw '포터블 실행 경로가 올바르지 않습니다.'
    }
    $sandboxPython = $portablePython
}
if (Test-Path -LiteralPath 'C:\wm_portable\active_python.txt') {
    # 준비 스크립트에서 import 검증을 통과한 독립 실행 환경을 우선한다.
    $portablePython = [IO.File]::ReadAllText('C:\wm_portable\active_python.txt').Trim()
    if ($portablePython -notmatch '^C:\\wm_portable\\[a-f0-9]{32}\\python\.exe$') {
        throw '포터블 실행 경로가 올바르지 않습니다.'
    }
    $sandboxPython = $portablePython
}
if (-not (Test-Path -LiteralPath $sandboxPython)) {
    $sandboxPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
}
if (-not (Test-Path -LiteralPath $sandboxPython)) { throw 'Python 준비를 먼저 완료하세요.' }

# 현재 폴더·사용자 사이트 모듈을 배제하고 승인된 검사 파일만 실행한다.
$entryScript = 'C:\wm_tools\inspect_pickle.py'
if ($Canonical) {
    # 구조 검사와 변환을 명시적으로 구분하고 변환을 자동 시작하지 않는다.
    $entryScript = 'C:\wm_tools\export_canonical.py'
}
& $sandboxPython -I $entryScript --input $InputFile --trusted-pickle
if ($LASTEXITCODE -ne 0) { throw '최초 검사에 실패했습니다. 상세 결과는 로컬에서만 확인하세요.' }
