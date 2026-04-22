import { Component, EventEmitter, Input, Output } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-confirmation-popup',
  imports: [TranslatePipe, FormsModule],
  templateUrl: './confirmation-popup.html',
  styleUrl: './confirmation-popup.css',
})
export class ConfirmationPopup {
  @Input({ required: true }) content!: string;
  @Input({ required: true }) icon!: string;
  @Input({ required: true }) iconStyle!: string;
  @Input({ required: true }) confirmButtonStyle!: string;
  @Input() detailLabel?: string;

  detailValue?: string;

  @Output() confirm: EventEmitter<void> = new EventEmitter();
  @Output() close: EventEmitter<void> = new EventEmitter();
  @Output() confirmWithSubmitValue: EventEmitter<string> = new EventEmitter();

  public closePopup(): void {
    this.close.emit();
  }

  public confirmPopup(): void {
    if (this.detailValue && this.detailLabel) {
      this.confirmWithSubmitValue.emit(this.detailValue);
    } else {
      this.confirm.emit();
    }
    this.close.emit();
  }
}
