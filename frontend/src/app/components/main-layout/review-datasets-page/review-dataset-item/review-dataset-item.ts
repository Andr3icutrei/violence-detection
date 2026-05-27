import {
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { DatasetWithVideosResponseDto } from '../../../../core/api/models/dataset-with-videos-response-dto';
import { FormArray, FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import {VideoResponseDto} from '../../../../core/api/models/video-response-dto';
import { DatasetResponseDto } from '../../../../core/api/models/dataset-response-dto';
import { FormSubmitDetail } from '../../../form-submit-detail/form-submit-detail';
import { Subscription } from 'rxjs';
import { DatasetUpdatedService } from '../../../../services/dataset_updated/dataset-updated.service';
import { ValidateModelResponseDto } from '../../../../core/api/models/validate-model-response-dto';
import { HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-review-dataset-item',
  imports: [TranslatePipe, ReactiveFormsModule, FormSubmitDetail],
  templateUrl: './review-dataset-item.html',
  styleUrl: './review-dataset-item.css',
})
export class ReviewDatasetItem implements OnInit, OnDestroy {
  @Input({ required: true }) datasetId!: number;
  dataset?: DatasetWithVideosResponseDto;
  isDatasetLoading: boolean = false;

  @Output() closeModal: EventEmitter<void> = new EventEmitter();

  isSubmitted: boolean = false;
  submitMessage: string | null = null;
  isSubmitSuccessful: boolean = false;

  isValidationInProgress: boolean = false;
  isValidationSuccessful: boolean = false;
  hasValidated: boolean = false;
  validationMessage: string | null = null;
  validationResult: ValidateModelResponseDto | null = null;

  form: FormGroup;

  datasetUpdatedSubscription!: Subscription;
  formChangesSubscription!: Subscription;

  public constructor(
    private datasetsService: DatasetsService,
    private formBuilder: FormBuilder,
    private cdr: ChangeDetectorRef,
    private datasetUpdatedService: DatasetUpdatedService,
  ) {
    this.form = formBuilder.group({
      reviewComment: ['', [Validators.required, Validators.maxLength(100)]],
      videoReviews: this.formBuilder.array([]),
    });
  }

  ngOnInit(): void {
    this.loadDataset();
    this.datasetUpdatedSubscription = this.datasetUpdatedService.connect().subscribe({
      next: () => {
        this.loadDataset();
      },
    });
    this.formChangesSubscription = this.videoReviews.valueChanges.subscribe(() => {
      if (this.hasValidated) {
        this.hasValidated = false;
        this.validationResult = null;
        this.validationMessage = null;
        this.isValidationSuccessful = false;
      }
    });
  }

  get videoReviews(): FormArray {
    return this.form.get('videoReviews') as FormArray;
  }

  public get areAllVideosExcluded(): boolean {
    if (!this.videoReviews?.controls) return false;
    return this.videoReviews.controls.length > 0 &&
           this.videoReviews.controls.every(c => c.get('isExcluded')?.value);
  }

  public populateVideoReviews(): void {
    if (!this.dataset?.videos) return;

    this.videoReviews.clear();
    this.dataset.videos.forEach((video: VideoResponseDto) => {
      const videoGroup = this.formBuilder.group({
        id: [video.id],
        isViolent: [false],
        isExcluded: [false],
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
        this.hasValidated = false;
        this.validationResult = null;
        this.validationMessage = null;
        this.isValidationSuccessful = false;
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
    const comment = (this.form.get('reviewComment')?.value as string | null) ?? '';
    const hasComment = comment.trim().length > 0;
    return hasComment && !this.isSubmitted && this.hasValidated;
  }

  private buildVideoReviewPayload(excludedIds: Set<number>): { video_id: number; is_violent: boolean }[] {
    return this.videoReviews.controls
      .filter((videoGroup) => !excludedIds.has(videoGroup.get('id')!.value as number))
      .map((videoGroup) => ({
        video_id: videoGroup.get('id')!.value as number,
        is_violent: videoGroup.get('isViolent')!.value as boolean,
      }));
  }

  private getExcludedVideoIds(): number[] {
    return this.videoReviews.controls
      .filter((videoGroup) => videoGroup.get('isExcluded')!.value as boolean)
      .map((videoGroup) => videoGroup.get('id')!.value as number);
  }

  public validateDataset(): void {
    if (this.isValidationInProgress) {
      return;
    }
    this.isValidationInProgress = true;
    this.validationMessage = null;
    this.isValidationSuccessful = false;
    this.validationResult = null;

    const excludedVideoIds = this.getExcludedVideoIds();
    const videos = this.buildVideoReviewPayload(new Set(excludedVideoIds));
    if (videos.length === 0) {
      this.validationMessage = 'review-datasets.validation-no-videos-message';
      this.isValidationSuccessful = false;
      this.hasValidated = false;
      this.isValidationInProgress = false;
      this.cdr.detectChanges();
      return;
    }

    this.datasetsService
      .validateDatasetModel(this.datasetId, videos, excludedVideoIds)
      .subscribe({
        next: (result) => {
          this.validationResult = result;
          this.validationMessage = 'review-datasets.validation-success-message';
          this.isValidationSuccessful = true;
          this.hasValidated = true;
          this.isValidationInProgress = false;
          this.cdr.detectChanges();
        },
        error: (error: HttpErrorResponse) => {
          this.validationMessage =
            error.status === 400
              ? 'review-datasets.validation-invalid-videos-message'
              : 'review-datasets.validation-failed-message';
          this.isValidationSuccessful = false;
          this.hasValidated = false;
          this.isValidationInProgress = false;
          this.cdr.detectChanges();
        },
      });
  }

  public reviewDataset(action: 'ACCEPT' | 'REJECT'): void {
    const excludedVideoIds = this.getExcludedVideoIds();
    const videos: { video_id: number; is_violent: boolean }[] = this.buildVideoReviewPayload(
      new Set(excludedVideoIds),
    );
    if (videos.length === 0) {
      this.isSubmitSuccessful = false;
      this.submitMessage = 'review-datasets.submit-no-videos-message';
      this.cdr.detectChanges();
      return;
    }
    const isApproved = action === 'ACCEPT';
    const comment = this.form.get('reviewComment')!.value as string;
    this.isSubmitted = true;
    this.datasetsService
      .reviewDataset(this.datasetId, isApproved, videos, comment, excludedVideoIds)
      .subscribe({
        next: (data: DatasetResponseDto) => {
          this.submitMessage = isApproved
            ? 'review-datasets.approval-success-message'
            : 'review-datasets.rejection-success-message';
          this.isSubmitSuccessful = true;
          this.datasetUpdatedService.emitUpdate();
          this.cdr.detectChanges();
        },
        error: (error) => {
          this.isSubmitSuccessful = false;
          if (error.status === 400) {
            this.submitMessage = 'review-datasets.submit-invalid-videos-message';
          } else if (error.status === 403) {
            this.submitMessage = 'review-datasets.permission-error-message';
          } else if (error.status === 404) {
            this.submitMessage = 'review-datasets.not-found-error-message';
          } else if (error.status === 500) {
            this.submitMessage = 'review-datasets.internal-server-error-message';
          }
          this.cdr.detectChanges();
        },
      });
  }

  ngOnDestroy(): void {
    this.datasetUpdatedService.disconnect();
    this.datasetUpdatedSubscription.unsubscribe();
    this.formChangesSubscription.unsubscribe();
  }
}
