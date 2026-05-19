import {
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
import { FormArray, FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import {TranslatePipe} from "@ngx-translate/core";
import { DatasetWithVideosResponseDto } from '../../../../core/api/models/dataset-with-videos-response-dto';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { DatasetResponseDto } from '../../../../core/api/models/dataset-response-dto';
import { HttpErrorResponse } from '@angular/common/http';
import { FormSubmitDetail } from '../../../form-submit-detail/form-submit-detail';
import { Subscription } from 'rxjs';
import { DatasetUpdatedService } from '../../../../services/dataset_updated/dataset-updated.service';
import { ValidateModelResponseDto } from '../../../../core/api/models/validate-model-response-dto';

@Component({
  selector: 'app-edit-dataset-item',
  imports: [ReactiveFormsModule, TranslatePipe, FormSubmitDetail],
  standalone: true,
  templateUrl: './edit-dataset-item.html',
  styleUrl: './edit-dataset-item.css',
})
export class EditDatasetItem implements OnInit, OnDestroy {
  @Input({ required: true }) datasetId!: number;
  originalDataset!: DatasetWithVideosResponseDto;
  currentDataset!: DatasetWithVideosResponseDto;
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
      videoReviews: this.formBuilder.array([]),
    });
  }

  ngOnInit(): void {
    this.loadCurrentDataset();
    this.datasetUpdatedSubscription = this.datasetUpdatedService.connect().subscribe({
      next: () => {
        this.loadCurrentDataset();
      },
    });
    this.formChangesSubscription = this.form.valueChanges.subscribe(() => {
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

  public populateVideoReviews(): void {
    if (!this.currentDataset?.videos) return;
    this.videoReviews.clear();
    for (let i = 0; i < this.currentDataset?.videos.length; i++) {
      const videoGroup = this.formBuilder.group({
        id: [this.currentDataset.videos[i].id],
        isViolent: [this.currentDataset.videos[i].is_violent],
      });
      this.videoReviews.push(videoGroup);
    }
  }

  public loadOriginalDataset(): void {
    this.originalDataset = {
      id: this.currentDataset.id,
      name: this.currentDataset.name,
      videos: [],
      status: this.currentDataset.status,
      is_official: this.currentDataset.is_official,
    } as DatasetWithVideosResponseDto;
    for(let i = 0; i < this.currentDataset?.videos.length; i++) {
      this.originalDataset.videos.push({
        id: this.currentDataset.videos[i].id,
        name: this.currentDataset.videos[i].name,
        dataset_id: 0,
        dataset_is_official: false,
        dataset_name: '',
        duration: 0,
        frame_rate: 0,
        path: '',
        uid: '',
        is_violent: this.currentDataset.videos[i].is_violent
      });
    }
  }

  public getInferenceModelLabel(): string | null {
    const modelPath = this.currentDataset?.inference_model_path;
    if (!modelPath) {
      return null;
    }
    const pathParts = modelPath.split(/[\\/]/);
    return pathParts[pathParts.length - 1] || modelPath;
  }

  public getInferenceModelName(): string | null {
    return this.currentDataset?.inference_model_name ?? null;
  }

  public loadCurrentDataset(): void {
    this.isDatasetLoading = true;
    this.datasetsService.getDatasetWithVideos(this.datasetId).subscribe({
      next: (dataset) => {
        this.currentDataset = dataset;
        this.populateVideoReviews();
        this.loadOriginalDataset();
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

  public close(): void {
    this.closeModal.emit();
  }

  public isFormValid(): boolean {
    return this.form.valid && !this.isSubmitted && this.isFormChanged() && this.hasValidated;
  }

  private isFormChanged(): boolean {
    if (!this.originalDataset?.videos || !this.videoReviews?.controls) {
      return false;
    }
    for (let i = 0; i < this.currentDataset.videos.length; i++) {
      const originalValue = this.originalDataset.videos[i].is_violent;
      const currentFormValue = this.videoReviews.controls[i].get('isViolent')?.value;

      if (originalValue !== currentFormValue) {
        return true;
      }
    }
    return false;
  }

  private buildVideoReviewPayload(): { video_id: number; is_violent: boolean }[] {
    return this.videoReviews.controls.map((videoGroup) => ({
      video_id: videoGroup.get('id')!.value as number,
      is_violent: videoGroup.get('isViolent')!.value as boolean,
    }));
  }

  public validateDataset(): void {
    if (this.isValidationInProgress) {
      return;
    }
    this.isValidationInProgress = true;
    this.validationMessage = null;
    this.isValidationSuccessful = false;
    this.validationResult = null;

    const videos = this.buildVideoReviewPayload();
    this.datasetsService.validateDatasetModel(this.datasetId, videos).subscribe({
      next: (result) => {
        this.validationResult = result;
        this.validationMessage = 'edit-dataset.validation-success-message';
        this.isValidationSuccessful = true;
        this.hasValidated = true;
        this.isValidationInProgress = false;
        this.cdr.detectChanges();
      },
      error: (error: HttpErrorResponse) => {
        this.validationMessage = 'edit-dataset.validation-failed-message';
        this.isValidationSuccessful = false;
        this.hasValidated = false;
        this.isValidationInProgress = false;
        this.cdr.detectChanges();
      },
    });
  }

  public editDataset(): void {
    const videos = this.buildVideoReviewPayload();
    this.isSubmitted = true;
    this.datasetsService.editDataset(this.datasetId, videos).subscribe({
      next: (data: DatasetResponseDto) => {
        this.submitMessage = 'edit-dataset.edit-success-message';
        this.isSubmitSuccessful = true;
        this.datasetUpdatedService.emitUpdate();
        this.cdr.detectChanges();
      },
      error: (error: HttpErrorResponse) => {
        this.isSubmitSuccessful = false;
        if (error.status === 400) {
          this.submitMessage = 'edit-dataset.videos-not-found-error';
        } else if (error.status === 403) {
          this.submitMessage = 'edit-dataset.permission-error';
        } else if (error.status === 404) {
          this.submitMessage = 'edit-dataset.dataset-not-found-error';
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
