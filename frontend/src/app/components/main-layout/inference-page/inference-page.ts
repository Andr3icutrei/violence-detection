import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { Video } from '../../../models/video.model';
import { VideosService } from '../../../services/videos/videos-service';
import { TranslatePipe } from '@ngx-translate/core';
import {
  AbstractControl,
  FormBuilder,
  FormGroup, FormsModule, ReactiveFormsModule,
  ValidationErrors,
  ValidatorFn,
} from '@angular/forms';

@Component({
  selector: 'app-inference-page',
  imports: [TranslatePipe, FormsModule, ReactiveFormsModule],
  templateUrl: './inference-page.html',
  styleUrl: './inference-page.css',
})
export class InferencePage implements OnInit {
  @ViewChild('leftVideo') leftVideoRef?: ElementRef<HTMLVideoElement>;
  @ViewChild('rightVideo') rightVideoRef?: ElementRef<HTMLVideoElement>;

  videoDetails: Video | null = null;

  inferenceForm: FormGroup;
  isInferenceSubmitted: boolean = false;

  minZoom: number = 1;
  maxZoom: number = 4;
  readonly zoomFactorPerTick: number = 1.12;
  scale: number = 1;
  offsetX: number = 0;
  offsetY: number = 0;

  isPlaying: boolean = false;
  durationSeconds: number = 0;
  currentTimeSeconds: number = 0;
  effectiveFrameRate: number = 0;
  readonly playbackRateOptions: number[] = [1, 0.75, 0.5, 0.25];
  playbackRate: number = 1;

  constructor(
    private readonly router: Router,
    private readonly fb: FormBuilder,
    private readonly videosService: VideosService,
  ) {
    this.inferenceForm = this.fb.group(
      {
        personTrackingSelected: [false],
        classificationSelected: [false],
      },
      { validators: this.requireAtLeastOneCheckbox() },
    );
  }

  ngOnInit(): void {
    const state = history.state as { videoDetails?: Partial<Video> } & Partial<Video>;
    const stateVideo = (state?.videoDetails ?? state) as Partial<Video> | null;

    if (!stateVideo || !stateVideo.uid) {
      this.redirectToDashboard();
      return;
    }

    this.videoDetails = stateVideo as Video;

    this.videosService.existsVideo(stateVideo.uid).subscribe({
      next: () => {},
      error: () => {
        this.redirectToDashboard();
      },
    });
  }

  public get datasetBadgeClass(): string {
    return this.videoDetails?.dataset?.is_official
      ? 'dataset-badge dataset-badge-official'
      : 'dataset-badge dataset-badge-unofficial';
  }

  public get datasetBadgeLabelKey(): string {
    return this.videoDetails?.dataset?.is_official
      ? 'inference.dataset-badge-official'
      : 'inference.dataset-badge-unofficial';
  }

  public formatDuration(seconds: number | null | undefined): string {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds) || seconds < 0) {
      return '--:--';
    }

    const totalSeconds = Math.floor(seconds);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  public formatFrameRate(frameRate: number | null | undefined): string {
    if (
      frameRate === null ||
      frameRate === undefined ||
      Number.isNaN(frameRate) ||
      frameRate <= 0
    ) {
      return '-';
    }

    return `${Number(frameRate.toFixed(2))} fps`;
  }

  private redirectToDashboard(): void {
    this.router.navigate(['/dashboard'], {
      queryParams: { returnUrl: '/inference' },
      replaceUrl: true,
    });
  }

  private requireAtLeastOneCheckbox(): ValidatorFn {
    return (control: AbstractControl): ValidationErrors | null => {
      const personTrackingSelected = control.get('personTrackingSelected')?.value;
      const classificationSelected = control.get('classificationSelected')?.value;

      if (!personTrackingSelected && !classificationSelected) {
        return { atLeastOneRequired: true };
      }

      if (personTrackingSelected || classificationSelected) {
        return null;
      }

      return { atLeastOneRequired: true };
    };
  }

  public submitInference(): void {

  }

  public isFormInvalid(): boolean {
    return this.inferenceForm.invalid  || this.isInferenceSubmitted;
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
  }

  public onVideoMetadataLoaded(): void {
    const leftVideo = this.leftVideoRef?.nativeElement;
    const rightVideo = this.rightVideoRef?.nativeElement;

    const metadataDuration = this.getPositiveNumber(leftVideo?.duration)
      ?? this.getPositiveNumber(rightVideo?.duration)
      ?? this.getPositiveNumber(this.videoDetails?.duration)
      ?? 0;

    const metadataFrameRate = this.getPositiveNumber(this.videoDetails?.frameRate) ?? 0;

    this.durationSeconds = metadataDuration;
    this.effectiveFrameRate = metadataFrameRate;
    this.currentTimeSeconds = this.clamp(this.currentTimeSeconds, 0, this.durationSeconds);
    this.applyPlaybackRateToVideos(this.playbackRate);

    this.seekTo(this.currentTimeSeconds);
  }

  public onPrimaryVideoTimeUpdate(): void {
    const leftVideo = this.leftVideoRef?.nativeElement;
    const rightVideo = this.rightVideoRef?.nativeElement;

    if (!leftVideo) {
      return;
    }

    this.currentTimeSeconds = leftVideo.currentTime;

    if (rightVideo && Math.abs(rightVideo.currentTime - leftVideo.currentTime) > 0.04) {
      rightVideo.currentTime = leftVideo.currentTime;
    }
  }

  public togglePlayback(): void {
    const leftVideo = this.leftVideoRef?.nativeElement;
    const rightVideo = this.rightVideoRef?.nativeElement;

    if (!leftVideo || !rightVideo) {
      return;
    }

    if (this.isPlaying) {
      leftVideo.pause();
      rightVideo.pause();
      this.isPlaying = false;
      return;
    }

    this.applyPlaybackRateToVideos(this.playbackRate);
    void leftVideo.play();
    void rightVideo.play();
    this.isPlaying = true;
  }

  public onPlaybackRateChange(event: Event): void {
    const target = event.target as HTMLSelectElement | null;
    if (!target) {
      return;
    }

    const requestedRate = Number(target.value);
    const clampedRate = this.clamp(requestedRate, 0.25, 1);
    this.playbackRate = this.playbackRateOptions.includes(clampedRate) ? clampedRate : 1;
    this.applyPlaybackRateToVideos(this.playbackRate);
  }

  public onPlaybackEnded(): void {
    this.isPlaying = false;
  }

  public seekFromSlider(event: Event): void {
    const target = event.target as HTMLInputElement | null;
    if (!target) {
      return;
    }

    this.seekTo(Number(target.value));
  }

  public seekTo(timeInSeconds: number): void {
    const leftVideo = this.leftVideoRef?.nativeElement;
    const rightVideo = this.rightVideoRef?.nativeElement;
    const targetTime = this.clamp(timeInSeconds, 0, this.durationSeconds || 0);

    if (leftVideo) {
      leftVideo.currentTime = targetTime;
    }

    if (rightVideo) {
      rightVideo.currentTime = targetTime;
    }

    this.currentTimeSeconds = targetTime;
  }

  public formatTimelineSeconds(seconds: number): string {
    if (!Number.isFinite(seconds)) {
      return '0.00s';
    }

    return `${seconds.toFixed(2)}s`;
  }

  public get currentFrame(): number {
    if (!this.effectiveFrameRate || this.effectiveFrameRate <= 0) {
      return 0;
    }

    return Math.floor(this.currentTimeSeconds * this.effectiveFrameRate);
  }


  private getPositiveNumber(value: number | null | undefined): number | null {
    if (value === null || value === undefined || Number.isNaN(value) || value <= 0) {
      return null;
    }

    return value;
  }

  private applyPlaybackRateToVideos(rate: number): void {
    const leftVideo = this.leftVideoRef?.nativeElement;
    const rightVideo = this.rightVideoRef?.nativeElement;

    if (leftVideo) {
      leftVideo.playbackRate = rate;
    }

    if (rightVideo) {
      rightVideo.playbackRate = rate;
    }
  }

  private getPanBounds(viewport: HTMLElement, scale: number): {
    minOffsetX: number;
    maxOffsetX: number;
    minOffsetY: number;
    maxOffsetY: number;
  } {
    if (scale <= this.minZoom) {
      return {
        minOffsetX: 0,
        maxOffsetX: 0,
        minOffsetY: 0,
        maxOffsetY: 0,
      };
    }

    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    const scaledWidth = width * scale;
    const scaledHeight = height * scale;

    return {
      minOffsetX: width - scaledWidth,
      maxOffsetX: 0,
      minOffsetY: height - scaledHeight,
      maxOffsetY: 0,
    };
  }

  private clampOffsets(viewport: HTMLElement, scale: number, offsetX: number, offsetY: number): {
    x: number;
    y: number;
  } {
    const bounds = this.getPanBounds(viewport, scale);

    return {
      x: this.clamp(offsetX, bounds.minOffsetX, bounds.maxOffsetX),
      y: this.clamp(offsetY, bounds.minOffsetY, bounds.maxOffsetY),
    };
  }

  public onVideosWheel(event: WheelEvent, viewport: HTMLElement): void {
    event.preventDefault();

    const rect = viewport.getBoundingClientRect();
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;

    const factor = event.deltaY < 0 ? this.zoomFactorPerTick : 1 / this.zoomFactorPerTick;
    const nextScale = this.clamp(this.scale * factor, this.minZoom, this.maxZoom);

    if (nextScale === this.scale) {
      return;
    }

    const contentX = (pointerX - this.offsetX) / this.scale;
    const contentY = (pointerY - this.offsetY) / this.scale;

    const rawOffsetX = pointerX - contentX * nextScale;
    const rawOffsetY = pointerY - contentY * nextScale;
    const clampedOffsets = this.clampOffsets(viewport, nextScale, rawOffsetX, rawOffsetY);

    this.scale = nextScale;
    this.offsetX = clampedOffsets.x;
    this.offsetY = clampedOffsets.y;

    if (this.scale === this.minZoom) {
      this.offsetX = 0;
      this.offsetY = 0;
    }
  }
}
