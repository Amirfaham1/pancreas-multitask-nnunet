param(
    [string]$WorkRoot = "D:\MLQuizWork",
    [ValidateSet("online", "offline", "disabled")]
    [string]$WandbMode = "offline",
    [int]$DataAugmentationProcesses = 2
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$resolvedWorkRoot = [System.IO.Path]::GetFullPath($WorkRoot)

$env:nnUNet_raw = Join-Path $resolvedWorkRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $resolvedWorkRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $resolvedWorkRoot "nnUNet_results"
$env:nnUNet_extTrainer = Join-Path $projectRoot "src"
$env:nnUNet_n_proc_DA = [string]$DataAugmentationProcesses
$env:nnUNet_compile = "false"
$env:PANCREAS_MT_EPOCHS = "200"
$env:PANCREAS_MT_TRAIN_ITERS = "125"
$env:PANCREAS_MT_VAL_ITERS = "30"
$env:PANCREAS_MT_CLS_LOSS_WEIGHT = "0.5"
$env:PANCREAS_MT_NONLESION_PATCH_WEIGHT = "0.25"
$env:PANCREAS_MT_LABEL_SMOOTHING = "0.05"
$env:PANCREAS_MT_OVERSAMPLE_FOREGROUND = "0.5"

if ($WandbMode -eq "disabled") {
    $env:nnUNet_wandb_enabled = "0"
    Remove-Item Env:nnUNet_wandb_mode -ErrorAction SilentlyContinue
} else {
    $env:nnUNet_wandb_enabled = "1"
    $env:nnUNet_wandb_project = "pancreas-multitask-amirfaham-fallahpour"
    $env:nnUNet_wandb_mode = $WandbMode
}

@{
    project_root = $projectRoot
    work_root = $resolvedWorkRoot
    nnUNet_raw = $env:nnUNet_raw
    nnUNet_preprocessed = $env:nnUNet_preprocessed
    nnUNet_results = $env:nnUNet_results
    nnUNet_extTrainer = $env:nnUNet_extTrainer
    wandb_mode = $WandbMode
    augmentation_processes = $DataAugmentationProcesses
}
