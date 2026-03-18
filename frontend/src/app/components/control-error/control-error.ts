import { Component, Input } from '@angular/core';
import { AbstractControl } from '@angular/forms';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-control-error',
  imports: [],
  standalone: true,
  templateUrl: './control-error.html',
  styleUrl: './control-error.css',
})
export class ControlError {
  @Input() control!: AbstractControl;

  constructor(private translationService: TranslateService) {

  }

  getErrorMessage(): string | null {
    if(this.control.invalid && this.control.dirty) {
      if (this.control.hasError('required'))
        return this.translationService.instant('error-messages.required');
      if(this.control.hasError('email'))
        return this.translationService.instant('error-messages.invalid-email');
      if(this.control.hasError('maxlength')) {
        const { requiredLength } = this.control.getError('maxlength');
        return this.translationService.instant('error-messages.max-length', { requiredLength });
      }
      if (this.control.hasError('minlength')) {
        const { requiredLength } = this.control.getError('minlength');
        return this.translationService.instant('error-messages.min-length', { requiredLength });
      }
      if(this.control.hasError('passwordMismatch')) {
        return this.translationService.instant('error-messages.password-mismatch');
      }
    }
    return null;
  }
}
