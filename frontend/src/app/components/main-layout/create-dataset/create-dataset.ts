import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
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

@Component({
  selector: 'app-create-dataset',
  imports: [TranslatePipe, ReactiveFormsModule, ControlError, FormSubmitDetail],
  templateUrl: './create-dataset.html',
  styleUrl: './create-dataset.css',
})
export class CreateDataset implements OnInit {
  form: FormGroup;

  selectedFiles: File[] = [];
  readonly maxFilesToUpload: number = 20;

  constructor(
    private formBuilder: FormBuilder,
    private cdr: ChangeDetectorRef,
    private datasetService: DatasetsService,
  ) {
    this.form = this.formBuilder.group({
      name: new FormControl(null, [Validators.required]),
    });
  }

  ngOnInit(): void {
    this.form.valueChanges.subscribe(() => {
      if (
        this.form.dirty &&
        (this.form.hasError('noFilesUploaded') ||
          this.form.hasError('tooManyFilesUploaded') ||
          this.form.hasError('differentFilesFormat'))
      ) {
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

    if (!input.files || input.files.length <= 0) {
      this.setNoFilesUploadedError(input);
      return;
    }

    if (input.files.length > this.maxFilesToUpload) {
      this.setTooManyFilesUploaded(input);
      return;
    }
    const filesArray = Array.from(input.files);
    const areAllFilesMP4 = filesArray.every((file) => {
      file.type === 'video/mp4';
    });

    if (!areAllFilesMP4) {
      this.setDifferentFilesFormat(input);
      return;
    }

    this.selectedFiles = filesArray;
  }

  public setNoFilesUploadedError(input: HTMLInputElement): void {
    this.form.setErrors({ noFilesUploaded: true });
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

  public submitDataset(): void {}
}
