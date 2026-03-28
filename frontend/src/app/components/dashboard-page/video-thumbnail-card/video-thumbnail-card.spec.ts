import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VideoThumbnailCard } from './video-thumbnail-card';

describe('VideoThumbnailCard', () => {
  let component: VideoThumbnailCard;
  let fixture: ComponentFixture<VideoThumbnailCard>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VideoThumbnailCard],
    }).compileComponents();

    fixture = TestBed.createComponent(VideoThumbnailCard);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
