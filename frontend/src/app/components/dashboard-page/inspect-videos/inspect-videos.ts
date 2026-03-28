import { ChangeDetectorRef, Component, OnInit, AfterViewInit, ElementRef, HostListener, ViewChild } from '@angular/core';
import { SearchBar } from '../../search-bar/search-bar';
import { VideosService } from '../../../services/videos/videos-service';
import { Dataset } from '../../../core/api/models/dataset';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { DatasetModel } from '../../../models/dataset.model';
import { DatasetResponseDto } from '../../../core/api/models/dataset-response-dto';
import { VideoResponseDto } from '../../../core/api/models/video-response-dto';
import { Video } from '../../../models/video.model';
import { VideoThumbnailCard } from '../video-thumbnail-card/video-thumbnail-card';
import { Paginator } from '../../paginator/paginator';
import { CommonModule } from '@angular/common';
import { TranslatePipe } from '@ngx-translate/core';

@Component({
  selector: 'app-inspect-videos',
  imports: [SearchBar, VideoThumbnailCard, Paginator, CommonModule, TranslatePipe],
  templateUrl: './inspect-videos.html',
  styleUrl: './inspect-videos.css',
})
export class InspectVideos implements OnInit, AfterViewInit {
  @ViewChild('videosGrid') videosGrid!: ElementRef;

  pageSize: number = 40;
  basePageSize: number = 40;
  page: number = 0;
  asc: boolean = true;
  searchTerm: string = '';
  datasets: DatasetModel[] = [];
  selectedDataset: Dataset = 0;
  videos: Video[] = [];
  hasMoreVideos: boolean = true;

  constructor(
    private cdr: ChangeDetectorRef,
    private videosService: VideosService,
    private datasetsService: DatasetsService,
  ) {}

  @HostListener('window:resize')
  onResize() {
    this.updatePageSize();
  }

  ngOnInit(): void {
    this.getDatasets();
  }

  ngAfterViewInit(): void {
    this.updatePageSize();
    this.getVideosPaged();
    this.cdr.detectChanges();
  }

  private updatePageSize(): void {
    if (this.videosGrid) {
      const containerWidth = this.videosGrid.nativeElement.clientWidth;
      if (containerWidth === 0) return;

      // Match the CSS limits: min-width: 18.75rem (300px), gap: 1.5rem (24px)
      const minItemWidth = 300;
      const gap = 24;

      let columns = Math.floor((containerWidth + gap) / (minItemWidth + gap));
      if (columns < 1) {
        columns = 1;
      }

      const targetRows = Math.ceil(this.basePageSize / columns);
      const newPageSize = targetRows * columns;

      if (newPageSize !== this.pageSize) {
        this.pageSize = newPageSize;
      }
    }
  }

  public getDatasets(): void {
    this.datasetsService.getDatasets().subscribe({
      next: (data: DatasetResponseDto[]): void => {
        this.datasets = data.map(
          (dataset: DatasetResponseDto): DatasetModel => ({
            id: dataset.id as Dataset,
            name: dataset.name,
          }),
        );
        this.cdr.detectChanges();
      },
      error: (err): void => {
        console.error('Failed to load topbar information.', err);
      },
    });
  }

  public getVideosPaged(): void {
    this.videosService
      .getVideosPaged(this.asc, this.page, this.pageSize, this.searchTerm, this.selectedDataset)
      .subscribe({
        next: (data: VideoResponseDto[]): void => {
          this.hasMoreVideos = data.length === this.pageSize;
          this.videos = data.map(
            (video: VideoResponseDto): Video => ({
              id: video.id,
              name: video.name,
              path: video.path,
              is_violent: video.is_violent,
              duration: video.duration,
              dataset: {
                id: video.dataset_id,
                name: video.dataset_name,
              } as DatasetModel,
            }),
          );
          this.cdr.detectChanges();
        },
        error: (err): void => {
          console.error('Failed to load videos.', err);
        },
      });
  }

  public onPageChange(newPage: number): void {
    this.page = newPage;
    this.getVideosPaged();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  public debouncedSearch(searchTerm: string): void {
    this.searchTerm = searchTerm;
    this.page = 0;
    this.getVideosPaged();
  }
}
