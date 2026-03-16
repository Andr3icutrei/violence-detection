import { Component, Input } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { PortalForm } from '../portal-form.type';

@Component({
  selector: 'app-form-description',
  imports: [TranslatePipe],
  templateUrl: './form-description.html',
  styleUrl: './form-description.css',
})
export class FormDescription {
  @Input() type!: PortalForm;
}
