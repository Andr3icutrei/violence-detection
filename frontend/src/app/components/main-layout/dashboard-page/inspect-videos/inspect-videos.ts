import { ChangeDetectorRef, Component, OnInit, AfterViewInit, ElementRef, HostListener, ViewChild } from '@angular/core';
import { SearchBar } from '../../../search-bar/search-bar';
import { VideosService } from '../../../../services/videos/videos-service';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { DatasetModel } from '../../../../models/dataset.model';
import { DatasetResponseDto } from '../../../../core/api/models/dataset-response-dto';
import { VideoResponseDto } from '../../../../core/api/models/video-response-dto';
import { Video } from '../../../../models/video.model';
import { VideoThumbnailCard } from '../video-thumbnail-card/video-thumbnail-card';
import { Paginator } from '../../../paginator/paginator';
import { CommonModule } from '@angular/common';
import { TranslatePipe } from '@ngx-translate/core';
import { CreateDataset } from '../../create-dataset/create-dataset';

@Component({
  selector: 'app-inspect-videos',
  imports: [SearchBar, VideoThumbnailCard, Paginator, CommonModule, TranslatePipe, CreateDataset],
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
  selectedDataset: DatasetModel | null = null;
  videos: Video[] = [];
  hasMoreVideos: boolean = true;
  isLoadingVideos: boolean = false;

  isCreateDatasetPopupOpen: boolean = false;

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
    this.datasetsService.getAcceptedDatasets().subscribe({
      next: (data: DatasetResponseDto[]): void => {
        this.datasets = data.map(
          (dataset: DatasetResponseDto): DatasetModel => ({
            id: dataset.id,
            name: dataset.name,
            is_official: dataset.is_official,
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
    this.isLoadingVideos = true;

    this.videosService
      .getVideosPaged(this.asc, this.page, this.pageSize, this.searchTerm, this.selectedDataset?.id)
      .subscribe({
        next: (data: VideoResponseDto[]): void => {
          this.hasMoreVideos = data.length === this.pageSize;
          this.videos = data.map(
            (video: VideoResponseDto): Video => ({
              id: video.id,
              uid: video.uid,
              name: video.name,
              path: video.path,
              is_violent: video.is_violent,
              duration: video.duration,
              frameRate: video.frame_rate,
              dataset: {
                id: video.dataset_id,
                name: video.dataset_name,
                is_official: video.dataset_is_official,
              } as DatasetModel,
            }),
          );
          this.isLoadingVideos = false;
          this.cdr.detectChanges();
        },
        error: (err): void => {
          console.error('Failed to load videos.', err);
          this.videos = [];
          this.hasMoreVideos = false;
          this.isLoadingVideos = false;
          this.cdr.detectChanges();
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

  public openCreateDatasetModal(): void {
    this.isCreateDatasetPopupOpen = true;
  }

  public closeCreateDatasetModal(): void {
    this.isCreateDatasetPopupOpen = false;
  }
}
