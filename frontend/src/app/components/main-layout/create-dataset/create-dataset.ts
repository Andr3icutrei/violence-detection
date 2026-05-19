import { ChangeDetectorRef, Component, DestroyRef, EventEmitter, OnInit, Output } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import {
  FormBuilder,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { ControlError } from '../../control-error/control-error';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { FormSubmitDetail } from '../../form-submit-detail/form-submit-detail';
import { HttpErrorResponse } from '@angular/common/http';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

@Component({
  selector: 'app-create-dataset',
  imports: [TranslatePipe, ReactiveFormsModule, ControlError, FormSubmitDetail],
  templateUrl: './create-dataset.html',
  styleUrl: './create-dataset.css',
})
export class CreateDataset implements OnInit {
  form: FormGroup;

  selectedFiles: File[] = [];
  selectedInferenceModel: File | null = null;
  readonly maxFilesToUpload: number = 20;
  readonly maxTotalUploadSizeBytes: number = 20 * 1024 * 1024;
  readonly maxUploadSizeMb: number = 20;
  readonly maxInferenceModelSizeBytes: number = 500 * 1024 * 1024;
  readonly maxInferenceModelSizeMb: number = 500;

  isSubmitted: boolean = false;
  submitStatusKey: string | null = null;
  success: boolean | null = null;
  submitMessageTranslateParams: Record<string, string | number> = {};
  @Output() closeModal = new EventEmitter<void>();

  private readonly formErrorPriority: string[] = [
    'noFilesUploaded',
    'tooManyFilesUploaded',
    'differentFilesFormat',
    'filesTooLarge',
    'modelInvalidFormat',
    'modelTooLarge',
  ];

  constructor(
    private formBuilder: FormBuilder,
    private cdr: ChangeDetectorRef,
    private datasetService: DatasetsService,
    private destroyRef: DestroyRef,
  ) {
    this.form = this.formBuilder.group({
      name: new FormControl(null, [Validators.required, Validators.maxLength(20)]),
    });
  }

  ngOnInit(): void {
    this.form.valueChanges
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(() => {
        if (
          this.form.dirty &&
          (this.form.hasError('noFilesUploaded') ||
            this.form.hasError('tooManyFilesUploaded') ||
            this.form.hasError('differentFilesFormat') ||
            this.form.hasError('filesTooLarge') ||
            this.form.hasError('modelInvalidFormat') ||
            this.form.hasError('modelTooLarge'))
        ) {
          this.submitStatusKey = null;
          this.submitMessageTranslateParams = {};
          this.success = null;
          this.form.setErrors(null);
          this.form.updateValueAndValidity({ emitEvent: false });
          this.cdr.detectChanges();
        }
      });
  }

  public isControlRequired(controlName: string): boolean {
    if (this.form.contains(controlName)) {
      const control = this.form.get(controlName);
      return control?.hasValidator(Validators.required) || false;
    }
    return false;
  }

  public onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFiles = [];
    this.submitStatusKey = null;
    this.success = null;
    this.submitMessageTranslateParams = {};
    this.form.setErrors(null);
    this.form.updateValueAndValidity({ emitEvent: false });

    if (!input.files || input.files.length <= 0) {
      this.setNoFilesUploadedError(input);
      return;
    }

    if (input.files.length > this.maxFilesToUpload) {
      this.setTooManyFilesUploaded(input);
      return;
    }
    const filesArray = Array.from(input.files);
    const areAllFilesMP4 = filesArray.every((file) => this.isMp4File(file));

    if (!areAllFilesMP4) {
      this.setDifferentFilesFormat(input);
      return;
    }

    const totalSize = filesArray.reduce((sum, file) => sum + file.size, 0);

    if (totalSize > this.maxTotalUploadSizeBytes) {
      this.setFilesTooLargeError(input);
      return;
    }

    this.selectedFiles = filesArray;
  }

  public onModelFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedInferenceModel = null;
    this.submitStatusKey = null;
    this.success = null;
    this.submitMessageTranslateParams = {};
    this.clearModelErrors();

    if (!input.files || input.files.length <= 0) {
      return;
    }

    const modelFile = input.files[0];
    if (!this.isOnnxFile(modelFile)) {
      this.setModelInvalidFormat(input);
      return;
    }

    if (modelFile.size > this.maxInferenceModelSizeBytes) {
      this.setModelTooLarge(input);
      return;
    }

    this.selectedInferenceModel = modelFile;
  }

  private isOnnxFile(file: File): boolean {
    return file.name.toLowerCase().endsWith('.onnx');
  }

  private setModelInvalidFormat(input: HTMLInputElement): void {
    this.setFormErrorKey('modelInvalidFormat');
    this.resetModelSelection(input);
  }

  private setModelTooLarge(input: HTMLInputElement): void {
    this.setFormErrorKey('modelTooLarge');
    this.resetModelSelection(input);
  }

  private resetModelSelection(inputElement: HTMLInputElement): void {
    this.selectedInferenceModel = null;
    inputElement.value = '';
  }

  private setFormErrorKey(errorKey: string): void {
    const errors = { ...(this.form.errors ?? {}) } as Record<string, boolean>;
    errors[errorKey] = true;
    this.form.setErrors(errors);
    this.cdr.detectChanges();
  }

  private clearModelErrors(): void {
    if (!this.form.errors) {
      return;
    }
    const errors = { ...(this.form.errors as Record<string, boolean>) };
    delete errors['modelInvalidFormat'];
    delete errors['modelTooLarge'];
    this.form.setErrors(Object.keys(errors).length > 0 ? errors : null);
    this.form.updateValueAndValidity({ emitEvent: false });
  }

  private isMp4File(file: File): boolean {
    return file.type === 'video/mp4' || file.name.toLowerCase().endsWith('.mp4');
  }

  public setNoFilesUploadedError(input: HTMLInputElement): void {
    this.form.setErrors({ noFilesUploaded: true });
    this.resetSelection(input);
    this.cdr.detectChanges();
  }

  public setFilesTooLargeError(input: HTMLInputElement): void {
    this.form.setErrors({ filesTooLarge: true });
    this.resetSelection(input);
    this.cdr.detectChanges();
  }

  public setTooManyFilesUploaded(input: HTMLInputElement): void {
    this.form.setErrors({ tooManyFilesUploaded: true });
    this.resetSelection(input);
    this.cdr.detectChanges();
  }

  public setDifferentFilesFormat(input: HTMLInputElement): void {
    this.form.setErrors({ differentFilesFormat: true });
    this.resetSelection(input);
    this.cdr.detectChanges();
  }

  resetSelection(inputElement: HTMLInputElement): void {
    this.selectedFiles = [];
    inputElement.value = '';
  }

  public isFormValid(): boolean {
    return this.form.valid && !this.isSubmitted && this.selectedFiles.length > 0;
  }

  public onContentClick(event: MouseEvent): void {
    event.stopPropagation();
  }

  public close(): void {
    this.closeModal.emit();
  }

  public getSubmitMessageKey(): string | null {
    if (this.success === true && this.submitStatusKey) {
      return `create-dataset.${this.toKebabCase(this.submitStatusKey)}`;
    }

    const errorKey = this.getActiveErrorKey();
    return errorKey ? `error-messages.${this.toKebabCase(errorKey)}` : null;
  }

  public getSubmitTranslateParams(): Record<string, string | number> {
    const errorKey = this.getActiveErrorKey();

    if (errorKey === 'tooManyFilesUploaded') {
      return { max: this.maxFilesToUpload };
    }
    if (errorKey === 'differentFilesFormat' || errorKey === 'invalidFileFormat') {
      return { format: '.mp4' };
    }
    if (errorKey === 'filesTooLarge') {
      return { max: this.maxUploadSizeMb };
    }
    if (errorKey === 'modelInvalidFormat') {
      return { format: '.onnx' };
    }
    if (errorKey === 'modelTooLarge') {
      return { max: this.maxInferenceModelSizeMb };
    }

    return this.submitMessageTranslateParams;
  }

  private getActiveErrorKey(): string | null {
    for (const errorKey of this.formErrorPriority) {
      if (this.form.hasError(errorKey)) {
        return errorKey;
      }
    }

    return this.success === false ? this.submitStatusKey : null;
  }

  private toKebabCase(value: string): string {
    return value.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
  }

  private resolveConflictErrorCode(error: HttpErrorResponse): string | null {
    const payload = error.error as unknown;

    if (!payload || typeof payload !== 'object') {
      return null;
    }

    const payloadRecord = payload as Record<string, unknown>;

    const detail = payloadRecord['detail'];
    if (typeof detail === 'string') {
      return detail;
    }
    if (detail && typeof detail === 'object') {
      const detailRecord = detail as Record<string, unknown>;
      if (typeof detailRecord['error_code'] === 'string') {
        return detailRecord['error_code'];
      }
    }

    if (typeof payloadRecord['error_code'] === 'string') {
      return payloadRecord['error_code'];
    }

    return null;
  }

  public submitDataset(): void {
    if (!this.isFormValid()) return;

    this.isSubmitted = true;

    const formData = new FormData();
    formData.append('name', this.form.get('name')?.value);

    for (const file of this.selectedFiles) {
      formData.append('videos', file, file.name);
    }

    if (this.selectedInferenceModel) {
      formData.append('inference_model', this.selectedInferenceModel, this.selectedInferenceModel.name);
    }

    this.datasetService.createUnofficialDataset(formData).subscribe({
      next: () => {
        this.isSubmitted = false;
        this.submitStatusKey = 'submitSuccessful';
        this.success = true;
        this.cdr.detectChanges();
      },
      error: (error: HttpErrorResponse) => {
        this.isSubmitted = false;
        this.success = false;
        if (error.status === 400) {
          this.submitStatusKey = 'invalidFileFormat';
        } else if (error.status === 404) {
          this.submitStatusKey = 'submitFailed';
        } else if (error.status === 409) {
          const conflictCode = this.resolveConflictErrorCode(error);
          if (conflictCode === 'DATASET_NAME_EXISTS') {
            this.submitStatusKey = 'datasetNameExists';
          } else if (conflictCode === 'USER_HAS_PENDING_DATASETS') {
            this.submitStatusKey = 'userHasPendingDatasets';
          } else {
            this.submitStatusKey = 'submitFailed';
          }
        } else if (error.status === 500) {
          this.submitStatusKey = 'submitFailed';
        } else {
          this.submitStatusKey = 'submitFailed';
        }
        this.cdr.detectChanges();
      },
    });
  }
}
