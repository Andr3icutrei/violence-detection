import { Component, EventEmitter, Input, Output } from '@angular/core';
import { UserResponseDto } from '../../../../core/api/models/user-response-dto';
import { UsersService } from '../../../../services/users/users-service';
import { TranslatePipe } from '@ngx-translate/core';
import { ConfirmationPopup } from '../../../confirmation-popup/confirmation-popup';

@Component({
  selector: 'app-users-table',
  imports: [TranslatePipe, ConfirmationPopup],
  templateUrl: './users-table.html',
  styleUrl: './users-table.css',
})
export class UsersTable {
  @Input({ required: true }) users: UserResponseDto[] = [];
  @Output() onUserChanged: EventEmitter<void> = new EventEmitter();

  isBanUserModalOpen: boolean = false;
  userToBan: UserResponseDto | null = null;
  banReason: string | null = null;

  constructor(private readonly usersService: UsersService) {}

  public getRoleTranslation(is_admin: boolean): string {
    return is_admin ? 'users.admin' : 'users.regular-user';
  }

  public getStatusTranslation(is_active: boolean, is_banned: boolean): string {
    if (!is_active) return 'users.inactive-account-user';
    return is_banned ? 'users.banned-user' : 'users.active-user';
  }

  public updateUserRole(user: UserResponseDto): void {
    this.usersService.updateUserRole(user.id!, !user.is_admin!).subscribe({
      next: () => {
        this.onUserChanged.emit();
      },
    });
  }

  public openBanModal(user: UserResponseDto): void {
    this.userToBan = user;
    this.isBanUserModalOpen = true;
  }

  public closeBanModal(): void {
    this.isBanUserModalOpen = false;
    this.userToBan = null;
  }

  public confirmBan(banReason: string): void {
    if (!this.userToBan) return;
    if (this.banReason) return;

    this.usersService.banUser(this.userToBan.id, banReason).subscribe({

    })
    this.closeBanModal();
    this.onUserChanged.emit();
  }
}
