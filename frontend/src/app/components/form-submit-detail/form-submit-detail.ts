import { Component, Input } from '@angular/core';
import { AbstractControl } from '@angular/forms';
import { TranslatePipe, TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-form-submit-detail',
  imports: [TranslatePipe],
  templateUrl: './form-submit-detail.html',
  styleUrl: './form-submit-detail.css',
})
export class FormSubmitDetail {
  @Input() message!: string;
  @Input() success!: boolean;

  constructor(private translationService: TranslateService) {

  }
}
