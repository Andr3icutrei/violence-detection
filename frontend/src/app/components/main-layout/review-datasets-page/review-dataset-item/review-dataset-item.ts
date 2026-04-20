import { ChangeDetectorRef, Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { DatasetToReviewResponseDto } from '../../../../core/api/models/dataset-to-review-response-dto';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { DatasetWithVideosResponseDto } from '../../../../core/api/models/dataset-with-videos-response-dto';
import { FormArray, FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import {VideoResponseDto} from '../../../../core/api/models/video-response-dto';
import { ControlError } from '../../../control-error/control-error';
import { DatasetResponseDto } from '../../../../core/api/models/dataset-response-dto';
import { FormSubmitDetail } from '../../../form-submit-detail/form-submit-detail';

@Component({
  selector: 'app-review-dataset-item',
  imports: [TranslatePipe, ReactiveFormsModule, FormSubmitDetail],
  templateUrl: './review-dataset-item.html',
  styleUrl: './review-dataset-item.css',
})
export class ReviewDatasetItem implements OnInit {
  @Input({ required: true }) datasetId!: number;
  dataset?: DatasetWithVideosResponseDto;
  isDatasetLoading: boolean = false;

  @Output() closeModal: EventEmitter<void> = new EventEmitter();

  isSubmitted: boolean = false;
  submitMessage: string | null = null;
  isSubmitSuccessful: boolean = false;

  form: FormGroup;

  public constructor(
    private datasetsService: DatasetsService,
    private formBuilder: FormBuilder,
    private cdr: ChangeDetectorRef,
  ) {
    this.form = formBuilder.group({
      reviewComment: ['', [Validators.required, Validators.maxLength(100)]],
      videoReviews: this.formBuilder.array([]),
    });
  }

  ngOnInit(): void {
    this.loadDataset();
  }

  get videoReviews(): FormArray {
    return this.form.get('videoReviews') as FormArray;
  }

  public populateVideoReviews(): void {
    if (!this.dataset?.videos) return;

    this.videoReviews.clear();
    this.dataset.videos.forEach((video: VideoResponseDto) => {
      const videoGroup = this.formBuilder.group({
        id: [video.id],
        isViolent: [false, Validators.required],
      });
      this.videoReviews.push(videoGroup);
    });
  }

  public loadDataset(): void {
    this.isDatasetLoading = true;
    this.datasetsService.getDatasetWithVideos(this.datasetId).subscribe({
      next: (dataset) => {
        this.dataset = dataset;
        this.populateVideoReviews();
        this.isDatasetLoading = false;
        this.cdr.detectChanges();
      },
      error: (error) => {
        this.closeModal.emit();
        this.isDatasetLoading = false;
      },
    });
  }

  public isControlRequired(controlName: string): boolean {
    return this.form.get(controlName)?.hasValidator(Validators.required) || false;
  }

  public close(): void {
    this.closeModal.emit();
  }

  public isFormValid(): boolean {
    return this.form.valid && !this.isSubmitted;
  }

  public reviewDataset(action: 'ACCEPT' | 'REJECT'): void {
    const videos: { video_id: number; is_violent: boolean }[] = this.videoReviews.controls.map(
      (videoGroup) => ({
        video_id: videoGroup.get('id')!.value as number,
        is_violent: videoGroup.get('isViolent')!.value as boolean,
      }),
    );
    const isApproved = action === 'ACCEPT';
    const comment = this.form.get('reviewComment')!.value as string;
    this.isSubmitted = true;
    this.datasetsService.reviewDataset(this.datasetId, isApproved, videos, comment).subscribe({
      next: (data: DatasetResponseDto) => {
        this.submitMessage = isApproved
          ? 'review-datasets.approval-success-message'
          : 'review-datasets.rejection-success-message';
        this.isSubmitSuccessful = true;
      },
      error: (error) => {
        this.isSubmitSuccessful = false;
        if (error.status === 403) {
          this.submitMessage = 'review-datasets.permission-error-message';
        } else if (error.status === 404) {
          this.submitMessage = 'review-datasets.not-found-error-message';
        } else if (error.status === 500) {
          this.submitMessage = 'review-datasets.internal-server-error-message';
        }
      },
    });
  }
}
