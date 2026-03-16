import { Component, Input } from '@angular/core';
import { AbstractControl } from '@angular/forms';
import { TranslatePipe, TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-form-error',
  imports: [TranslatePipe],
  templateUrl: './form-error.html',
  styleUrl: './form-error.css',
})
export class FormError {
  @Input() errorMessage!: string;

  constructor(private translationService: TranslateService) {}
}
