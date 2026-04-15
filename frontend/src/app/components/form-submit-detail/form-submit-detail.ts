import { Component, Input } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';

@Component({
  selector: 'app-form-submit-detail',
  imports: [TranslatePipe],
  templateUrl: './form-submit-detail.html',
  styleUrl: './form-submit-detail.css',
})
export class FormSubmitDetail {
  @Input() message!: string;
  @Input() success!: boolean;
  @Input() translateParams?: Record<string, string | number>;
}
