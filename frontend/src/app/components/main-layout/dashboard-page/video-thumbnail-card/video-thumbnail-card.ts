import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Video } from '../../../../models/video.model';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-video-thumbnail-card',
  standalone: true,
  imports: [CommonModule, TranslatePipe],
  templateUrl: './video-thumbnail-card.html',
  styleUrl: './video-thumbnail-card.css',
})
export class VideoThumbnailCard {
  @Input({ required: true }) video!: Video;

  constructor(private router:Router) {

  }

  public formatDuration(seconds: number): string {
    if (!seconds) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  public playPreview(videoElement: HTMLVideoElement): void {
    videoElement.play().catch(() => {});
  }

  public stopPreview(videoElement: HTMLVideoElement): void {
    videoElement.pause();
    videoElement.currentTime = 0;
  }

  public goToInferencePage(): void {
    this.router.navigate(['/inference'], {
      state: {
        videoDetails: this.video,
        fromVideoCard: true,
      }
    });
  }
}
