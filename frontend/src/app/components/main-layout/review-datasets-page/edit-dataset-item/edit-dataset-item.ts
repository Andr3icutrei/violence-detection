import {
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
import { FormArray, FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import {TranslatePipe} from "@ngx-translate/core";
import { DatasetWithVideosResponseDto } from '../../../../core/api/models/dataset-with-videos-response-dto';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { DatasetResponseDto } from '../../../../core/api/models/dataset-response-dto';
import { HttpErrorResponse } from '@angular/common/http';
import { FormSubmitDetail } from '../../../form-submit-detail/form-submit-detail';
import { Subscription } from 'rxjs';
import { DatasetUpdatedService } from '../../../../services/dataset-updated/dataset-updated.service';
import {DatasetStatusModel} from '../../../../models/dataset-status.model';

@Component({
  selector: 'app-edit-dataset-item',
  imports: [ReactiveFormsModule, TranslatePipe, FormSubmitDetail],
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

  form: FormGroup;

  datasetUpdatedSubscription!: Subscription;

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
        isViolent: [this.currentDataset.videos[i].is_violent, Validators.required],
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

  public loadCurrentDataset(): void {
    this.isDatasetLoading = true;
    this.datasetsService.getDatasetWithVideos(this.datasetId).subscribe({
      next: (dataset) => {
        this.currentDataset = dataset;
        this.populateVideoReviews();
        this.loadOriginalDataset();
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
    return this.form.valid && !this.isSubmitted && this.isFormChanged();
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

  public editDataset(): void {
    const videos: { video_id: number; is_violent: boolean }[] = this.videoReviews.controls.map(
      (videoGroup) => ({
        video_id: videoGroup.get('id')!.value as number,
        is_violent: videoGroup.get('isViolent')!.value as boolean,
      }),
    );
    this.isSubmitted = true;
    this.datasetsService.editDataset(this.datasetId, videos).subscribe({
      next: (data: DatasetResponseDto) => {
        this.submitMessage = 'edit-dataset.edit-success-message';
        this.isSubmitSuccessful = true;
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
  }
}
