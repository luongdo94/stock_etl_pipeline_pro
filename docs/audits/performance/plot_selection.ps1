# Existing .NET charting only. No packages, providers, databases or production edits.
# This plots the JavaScript selection analogue, NOT measured Python application speed.
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
$perfData = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'selection_results.json') -Raw | ConvertFrom-Json
$perfChart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$perfChart.Width = 1100
$perfChart.Height = 650
$perfChart.BackColor = [System.Drawing.Color]::White
$perfArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea 'Scaling'
$perfArea.AxisX.IsLogarithmic = $true
$perfArea.AxisY.IsLogarithmic = $true
$perfArea.AxisX.Title = 'Companies (756 price observations each; logarithmic scale)'
$perfArea.AxisY.Title = 'Median selection time, ms (logarithmic scale)'
$perfArea.AxisX.Minimum = 1
$perfArea.AxisX.Maximum = 6000
$perfArea.AxisY.Minimum = 0.001
$perfArea.AxisY.Maximum = 1000
$perfArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
$perfArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
$perfChart.ChartAreas.Add($perfArea)
$perfChart.Legends.Add((New-Object System.Windows.Forms.DataVisualization.Charting.Legend))
foreach ($perfVariant in @('baseline', 'partitioned')) {
    $perfSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series $perfVariant
    $perfSeries.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
    $perfSeries.BorderWidth = 3
    $perfSeries.MarkerStyle = [System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Circle
    $perfSeries.MarkerSize = 8
    foreach ($perfRow in $perfData.results) {
        if ($perfRow.variant -ne $perfVariant -or $perfRow.skipped) { continue }
        $perfTimes = @($perfRow.trials | ForEach-Object { $_.wallMs } | Sort-Object)
        $perfMedian = $perfTimes[[int][Math]::Floor($perfTimes.Count / 2)]
        [void]$perfSeries.Points.AddXY([double]$perfRow.n, [double]$perfMedian)
    }
    $perfChart.Series.Add($perfSeries)
}
[void]$perfChart.Titles.Add('Isolated JavaScript selection experiment — NOT Python / app performance')
[void]$perfChart.Titles.Add('5 timed calls per point; small-N JIT/timer noise; baseline at 5,000 not run')
$perfChart.SaveImage((Join-Path $PSScriptRoot 'selection_scaling.png'),
    [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$perfChart.Dispose()
