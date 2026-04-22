import { ChangeDetectorRef, Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import {DatasetItem} from "../dataset-item/dataset-item";
import {Paginator} from "../../../paginator/paginator";
import { FormArray, FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { SearchBar } from '../../../search-bar/search-bar';
import {TitleCasePipe} from "@angular/common";
import {TranslatePipe} from "@ngx-translate/core";
import { DatasetWithVideosResponseDto } from '../../../../core/api/models/dataset-with-videos-response-dto';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { VideoResponseDto } from '../../../../core/api/models/video-response-dto';
import { DatasetResponseDto } from '../../../../core/api/models/dataset-response-dto';
import { HttpErrorResponse } from '@angular/common/http';
import { FormSubmitDetail } from '../../../form-submit-detail/form-submit-detail';

@Component({
  selector: 'app-edit-dataset-item',
  imports: [ReactiveFormsModule, TranslatePipe, FormSubmitDetail],
  templateUrl: './edit-dataset-item.html',
  styleUrl: './edit-dataset-item.css',
})
export class EditDatasetItem implements OnInit {
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
    for (let i = 0; i < this.dataset?.videos.length; i++) {
      const videoGroup = this.formBuilder.group({
        id: [this.dataset.videos[i].id],
        isViolent: [this.dataset.videos[i].is_violent, Validators.required],
      });
      this.videoReviews.push(videoGroup);
    }
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

  public close(): void {
    this.closeModal.emit();
  }

  public isFormValid(): boolean {
    return this.form.valid && !this.isSubmitted;
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
}
